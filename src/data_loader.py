import os
import shutil
import zipfile
import glob as _glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


def _download_dataset(data_root: str):
    """自动下载数据集（已存在则跳过）。"""
    import kagglehub

    train_dir = os.path.join(data_root, "train")
    if os.path.isdir(train_dir) and os.listdir(train_dir):
        return  # 已存在，跳过

    print(f"数据集不存在，开始自动下载到 {data_root}...")
    download_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"下载完成: {download_path}")

    # kagglehub 下载后是原始压缩包或解压目录，需要复制到 data_root
    # 先尝试找直接解压好的目录结构
    extracted_dir = os.path.join(download_path, "chest_xray")
    if os.path.isdir(extracted_dir):
        # 直接是解压好的 chest_xray 目录
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(data_root, item)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    else:
        # 可能是 zip 文件，需要解压
        zip_files = list(_glob.glob(os.path.join(download_path, "*.zip")))
        if zip_files:
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                # 检查 zip 内是否包含 chest_xray 前缀
                members = zf.namelist()
                for member in members:
                    if member.startswith("chest_xray/"):
                        rel = member[len("chest_xray/"):]
                        if not rel:
                            continue
                        target = os.path.join(data_root, rel)
                        if member.endswith("/"):
                            os.makedirs(target, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with zf.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
        else:
            shutil.copytree(download_path, data_root, dirs_exist_ok=True)

    print(f"数据集已准备就绪: {data_root}")


def get_dataloaders(config: dict):
    """根据配置构建训练/验证/测试 DataLoader。"""
    data_root = config["data"]["data_root"]
    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    # 自动下载数据集（已存在则跳过）
    _download_dataset(data_root)

    # 检查数据目录
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"训练数据目录不存在: {train_dir}\n"
            f"自动下载失败，请手动从 Kaggle 下载：\n"
            f"https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            f"解压到 {data_root}/ 目录下。"
        )

    # ImageNet 均值与标准差（预训练模型标准化）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    aug_cfg = config.get("augmentation", {})
    hflip_prob = aug_cfg.get("horizontal_flip_prob", 0.5)
    rot_degrees = aug_cfg.get("rotation_degrees", 15)
    brightness = aug_cfg.get("brightness_range", [0.8, 1.2])

    # 训练集增强
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomRotation(degrees=rot_degrees),
        transforms.ColorJitter(brightness=brightness),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # 验证集/测试集预处理（无随机增强）
    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    print(f"训练集类别映射: {train_dataset.class_to_idx}")
    print(f"训练集样本数: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    loaders = {"train": train_loader}

    if os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transforms)
        print(f"验证集样本数: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        loaders["val"] = val_loader

    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transforms)
        print(f"测试集样本数: {len(test_dataset)}")
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        loaders["test"] = test_loader

    return loaders
