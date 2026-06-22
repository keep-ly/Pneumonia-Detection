import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置随机种子，确保训练可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(train_dir: str) -> float:
    """根据训练集正负样本比例计算 pos_weight（肺炎为正类）。
    肺炎样本数记为 pos，正常样本数记为 neg，pos_weight = neg / pos。
    """
    normal_dir = os.path.join(train_dir, "NORMAL")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
    num_normal = len([f for f in os.listdir(normal_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    num_pneumonia = len([f for f in os.listdir(pneumonia_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if num_pneumonia == 0:
        return 1.0
    pos_weight = num_normal / num_pneumonia
    print(f"训练集统计: NORMAL={num_normal}, PNEUMONIA={num_pneumonia}, pos_weight={pos_weight:.4f}")
    return pos_weight


class EarlyStopping:
    """早停机制：监控验证损失，若在 patience 个 epoch 内无改善则停止训练。"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """计算二分类指标（肺炎为正类）。"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics
