import torch
import numpy as np
import json
import pickle
import os
import logging

# ==================== 日志配置 ====================
def setup_logger():
    logger = logging.getLogger("ca_e2e_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    file_handler = logging.FileHandler("ca_e2e_train.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# ==================== 超参数 ====================
CA_GRID = 128
CA_STEPS = 15
RULE_ID = 30
_RESULTS_BASE = "results/classification_results/machine_learning/ca"
TEMPLATE_PATH = f"{_RESULTS_BASE}/ca_class_templates.pkl"
RESULT_PATH = f"{_RESULTS_BASE}/ca_e2e_train_result.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SENTINEL = object()

# ==================== 元胞自动机核心（纯CA） ====================
def rule30(neigh):
    rule = {0b111:0, 0b110:0, 0b101:0, 0b100:1,
            0b011:1, 0b010:1, 0b001:1, 0b000:0}
    return rule[int("".join(map(str, neigh)), 2)]

def vec2ca(feat, grid_size):
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    grid = np.interp(np.linspace(0, len(feat)-1, grid_size),
                     np.arange(len(feat)), feat)
    return (grid > 0.5).astype(np.int32)

def ca_run(grid, steps):
    cur = grid.copy()
    L = len(cur)
    for _ in range(steps):
        nxt = np.zeros_like(cur)
        for i in range(L):
            Lc = cur[i-1] if i>0 else 0
            Cc = cur[i]
            Rc = cur[i+1] if i<L-1 else 0
            nxt[i] = rule30([Lc, Cc, Rc])
        cur = nxt
    return cur

def get_ca_signature(steady_grid):
    """CA 稳态特征签名（纯结构特征，端到端核心）"""
    d1 = np.mean(steady_grid)
    d2 = np.mean(np.abs(np.diff(steady_grid)))
    d3 = np.sum(steady_grid) / len(steady_grid)
    return np.array([d1, d2, d3], dtype=np.float32)

# ==================== 训练：生成类别模板 ====================
def train_ca_classifier(loader, template_save_path=_SENTINEL, result_save_path=_SENTINEL):
    """
    :param loader: DataLoader
    :param template_save_path: 模板保存路径，None 则不保存，不传则用默认
    :param result_save_path: 训练结果保存路径，None 则不保存，不传则用默认
    """
    tpl_path = TEMPLATE_PATH if template_save_path is _SENTINEL else template_save_path
    res_path = RESULT_PATH if result_save_path is _SENTINEL else result_save_path
    try:
        logger.info("开始纯CA端到端分类训练")
        class_features = {}
        all_labels = []

        for data, label in loader:
            data = data.cpu().numpy().reshape(data.shape[0], -1)
            label = label.cpu().numpy()

            for x, y in zip(data, label):
                ca_grid = vec2ca(x, CA_GRID)
                steady = ca_run(ca_grid, CA_STEPS)
                sig = get_ca_signature(steady)

                if y not in class_features:
                    class_features[y] = []
                class_features[y].append(sig)
                all_labels.append(y)

        # 生成每个类的平均模板
        templates = {}
        for c, feats in class_features.items():
            templates[int(c)] = np.mean(feats, axis=0).tolist()

        n_class = len(templates)
        logger.info(f"训练完成，共 {n_class} 个类别模板")

        result = {
            "n_class": n_class,
            "n_samples": len(all_labels),
            "ca_grid": CA_GRID,
            "ca_steps": CA_STEPS,
            "rule": RULE_ID,
            "class_templates": templates
        }

        # 保存（路径为 None 则不保存）
        if tpl_path is not None:
            with open(tpl_path, "wb") as f:
                pickle.dump({
                    "templates": templates,
                    "grid": CA_GRID,
                    "steps": CA_STEPS
                }, f)
            logger.info(f"模板已保存：{tpl_path}")
        if res_path is not None:
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            logger.info(f"训练结果已保存：{res_path}")
        return result

    except Exception as e:
        logger.error("训练失败", exc_info=True)
        raise

# ==================== 测试入口 ====================
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    class MockSet(Dataset):
        def __len__(self): return 800
        def __getitem__(self, i):
            return torch.randn(16), torch.randint(0,5,()).item()

    dl = DataLoader(MockSet(), batch_size=32, shuffle=True)
    train_ca_classifier(dl)