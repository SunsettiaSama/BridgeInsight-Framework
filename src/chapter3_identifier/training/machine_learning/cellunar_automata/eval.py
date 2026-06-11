import torch
import numpy as np
import json
import pickle
import os
import logging

# ==================== 日志配置 ====================
def setup_logger():
    logger = logging.getLogger("ca_e2e_infer")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    file_handler = logging.FileHandler("ca_e2e_infer.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# ==================== 路径 ====================
_RESULTS_BASE = "results/classification_results/machine_learning/ca"
TEMPLATE_PATH = f"{_RESULTS_BASE}/ca_class_templates.pkl"
INFER_RESULT = f"{_RESULTS_BASE}/ca_e2e_infer_result.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SENTINEL = object()

# ==================== 纯CA核心（与训练完全一致） ====================
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
    d1 = np.mean(steady_grid)
    d2 = np.mean(np.abs(np.diff(steady_grid)))
    d3 = np.sum(steady_grid) / len(steady_grid)
    return np.array([d1, d2, d3], dtype=np.float32)

# ==================== 纯CA推理：模板匹配 ====================
def infer_ca_classifier(loader, has_label=True, template_load_path=_SENTINEL, infer_result_path=_SENTINEL):
    """
    :param loader: DataLoader
    :param has_label: 是否有标签
    :param template_load_path: 模板加载路径，不传则用默认
    :param infer_result_path: 推理结果保存路径，None 则不保存，不传则用默认
    """
    tpl_path = TEMPLATE_PATH if template_load_path is _SENTINEL else template_load_path
    res_path = INFER_RESULT if infer_result_path is _SENTINEL else infer_result_path
    try:
        logger.info("加载CA分类模板")
        with open(tpl_path, "rb") as f:
            pkg = pickle.load(f)

        templates = pkg["templates"]
        grid_size = pkg["grid"]
        steps = pkg["steps"]

        preds = []
        trues = []
        sample_ids = []

        logger.info("开始纯CA端到端推理")
        for bid, batch in enumerate(loader):
            if has_label:
                data, lbl = batch
                lbl = lbl.cpu().numpy()
                trues.extend(lbl.tolist())
            else:
                data = batch

            data = data.cpu().numpy().reshape(data.shape[0], -1)
            bs = data.shape[0]

            for i, x in enumerate(data):
                ca_g = vec2ca(x, grid_size)
                steady = ca_run(ca_g, steps)
                sig = get_ca_signature(steady)

                # 纯CA匹配：找最近的类别模板
                min_dist = np.inf
                best_cls = -1
                for c, t in templates.items():
                    t = np.array(t)
                    dist = np.linalg.norm(sig - t)
                    if dist < min_dist:
                        min_dist = dist
                        best_cls = c

                preds.append(best_cls)
                sample_ids.append(f"b{bid}_s{i}")

        # 构建结果
        res = {
            "total": len(sample_ids),
            "predictions": []
        }

        for i, sid in enumerate(sample_ids):
            item = {"id": sid, "pred": preds[i]}
            if has_label:
                item["true"] = trues[i]
            res["predictions"].append(item)

        # 准确率
        if has_label:
            acc = np.mean(np.array(preds) == np.array(trues))
            res["accuracy"] = float(acc)
            logger.info(f"推理准确率: {acc:.4f}")

        # 保存（路径为 None 则不保存）
        if res_path is not None:
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4)
            logger.info(f"推理结果已保存：{res_path}")
        return res

    except Exception as e:
        logger.error("推理失败", exc_info=True)
        raise

# ==================== 测试入口 ====================
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    class MockTestSet(Dataset):
        def __len__(self): return 200
        def __getitem__(self, i):
            return torch.randn(16), torch.randint(0,5,()).item()

    dl = DataLoader(MockTestSet(), batch_size=32, shuffle=False)
    infer_ca_classifier(dl, has_label=True)