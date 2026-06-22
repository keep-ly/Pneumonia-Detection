import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from model import create_model
from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="肺炎检测模型评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--data_split", type=str, default="val", choices=["val", "test"], help="评估数据集")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """在指定数据集上运行推理，返回标签、预测和概率。"""
    model.eval()
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="评估中")
    for images, labels in pbar:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_labels.extend(labels.numpy().flatten().tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    """绘制并保存 ROC 曲线。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Pneumonia Detection", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC 曲线已保存: {save_path}")


def plot_confusion_matrix(cm, save_path):
    """绘制并保存混淆矩阵热力图。"""
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
        cbar=True
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_loss_curve(history_path, save_path):
    """绘制训练/验证损失曲线（如果有 history）。"""
    if not os.path.exists(history_path):
        return

    import json
    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"损失/准确率曲线已保存: {save_path}")


def main():
    args = parse_args()

    # 加载配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 将相对路径转为基于项目根目录的绝对路径
    for key in ["data_root"]:
        val = config["data"][key]
        if not os.path.isabs(val):
            config["data"][key] = os.path.normpath(os.path.join(project_root, val))
    for key in ["log_dir", "checkpoint_dir", "result_dir"]:
        val = config["logging"][key]
        if not os.path.isabs(val):
            config["logging"][key] = os.path.normpath(os.path.join(project_root, val))

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("config", config)
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载模型: {args.checkpoint}")

    # 加载数据
    loaders = get_dataloaders(config)
    if args.data_split not in loaders:
        print(f"错误: 数据集 '{args.data_split}' 不存在，可用: {list(loaders.keys())}")
        sys.exit(1)
    loader = loaders[args.data_split]
    print(f"评估数据集: {args.data_split}, 样本数: {len(loader.dataset)}")

    # 推理
    y_true, y_prob = evaluate(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = np.array(metrics.pop("confusion_matrix"))

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"\n混淆矩阵:\n{cm}")

    # 保存可视化
    result_dir = config["logging"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    split_name = args.data_split
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"],
                   os.path.join(result_dir, f"roc_curve_{split_name}.png"))
    plot_confusion_matrix(cm,
                          os.path.join(result_dir, f"confusion_matrix_{split_name}.png"))

    # 绘制训练曲线（从 TensorBoard JSON 历史文件读取）
    history_path = os.path.join(config["logging"]["log_dir"], "training_history.json")
    loss_curve_path = os.path.join(result_dir, "loss_accuracy_curve.png")
    plot_loss_curve(history_path, loss_curve_path)

    # 保存指标到文本文件
    metrics_path = os.path.join(result_dir, f"metrics_{split_name}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"评估数据集: {split_name}\n")
        f.write(f"样本数: {len(y_true)}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
        f.write(f"\n混淆矩阵:\n{cm}\n")
    print(f"指标文件已保存: {metrics_path}")

    print("\n评估完成！")


if __name__ == "__main__":
    main()
