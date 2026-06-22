import os
import sys
import argparse
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from model import create_model
from utils import set_seed, compute_pos_weight, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description="肺炎检测模型训练")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, writer, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().flatten().tolist())
        all_probs.extend(probs.cpu().numpy().flatten().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # 计算 AUC
    from sklearn.metrics import roc_auc_score
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    return epoch_loss, epoch_acc, auc


def train(config_path: str, resume_path: str = None):
    # 1. 加载配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 将相对路径转为基于项目根目录的绝对路径，确保不受 cwd 影响
    for section, keys in [
        ("data", ["data_root"]),
        ("logging", ["log_dir", "checkpoint_dir", "result_dir"]),
    ]:
        for key in keys:
            val = config[section][key]
            if not os.path.isabs(val):
                config[section][key] = os.path.normpath(os.path.join(project_root, val))

    # 2. 设置随机种子
    set_seed(42)

    # 3. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 数据加载
    loaders = get_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    if val_loader is None:
        print("警告: 未找到验证集，将使用训练集作为近似验证集（不推荐）")
        val_loader = train_loader

    # 5. 损失函数
    data_root = config["data"]["data_root"]
    pos_weight_cfg = config["training"].get("pos_weight", "auto")
    if pos_weight_cfg == "auto":
        pos_weight = compute_pos_weight(os.path.join(data_root, "train"))
    else:
        pos_weight = float(pos_weight_cfg)
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"损失函数: BCEWithLogitsLoss, pos_weight={pos_weight:.4f}")

    # 6. 模型
    model = create_model(config).to(device)
    print(f"模型: {config['model']['architecture']}, 预训练={config['model']['pretrained']}")

    # 7. 优化器与调度器
    opt_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"])
    )

    epochs = int(opt_cfg["epochs"])
    lr_scheduler_cfg = opt_cfg.get("lr_scheduler", "cosine")
    if lr_scheduler_cfg == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler_cfg == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    else:
        scheduler = None

    # 8. 混合精度
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 9. 日志与检查点
    log_cfg = config["logging"]
    os.makedirs(log_cfg["log_dir"], exist_ok=True)
    os.makedirs(log_cfg["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=log_cfg["log_dir"])

    # 10. 早停
    patience = int(opt_cfg.get("early_stopping_patience", 7))
    early_stopping = EarlyStopping(patience=patience)

    start_epoch = 0
    best_val_loss = float("inf")

    # 恢复训练
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"从 {resume_path} 恢复训练，起始 epoch: {start_epoch}")

    # 11. 训练循环
    print(f"\n开始训练: {epochs} epochs, 早停 patience={patience}")
    print("-" * 60)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch + 1, scaler
        )

        # 验证
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device, writer, epoch + 1
        )

        # 日志
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  |  Val AUC: {val_auc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Acc/train", train_acc, epoch + 1)
        writer.add_scalar("Acc/val", val_acc, epoch + 1)
        writer.add_scalar("AUC/val", val_auc, epoch + 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # 学习率调度
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(log_cfg["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, checkpoint_path)
            print(f"  -> 保存最佳模型: {checkpoint_path}")

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n早停触发！最佳验证损失: {best_val_loss:.4f}")
            break

    # 保存训练历史为 JSON
    history_path = os.path.join(log_cfg["log_dir"], "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存: {history_path}")

    writer.close()
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存于: {os.path.join(log_cfg['checkpoint_dir'], 'best_model.pth')}")
    print(f"TensorBoard 日志: {log_cfg['log_dir']}")


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.resume)
