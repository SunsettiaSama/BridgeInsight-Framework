import torch
import numpy as np
import json
import pickle
import os
import logging
from scipy.stats import entropy
from sklearn.metrics import accuracy_score

# -------------------------- 日志配置 --------------------------
def setup_logger():
    logger = logging.getLogger("ca_nb_infer")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("ca_nb_infer.log", encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# -------------------------- 配置路径 --------------------------
_RESULTS_BASE = "results/classification_results/machine_learning/ca_bayes"
CA_PARAMS_PATH = f"{_RESULTS_BASE}/ca_params.pkl"
NB_MODEL_PATH = f"{_RESULTS_BASE}/ca_nb_model.pkl"
INFER_RESULT_PATH = f"{_RESULTS_BASE}/ca_nb_infer_result.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SENTINEL = object()

# -------------------------- CA 核心函数（与训练完全一致） --------------------------
def rule30(neighborhood):
    rule_map = {
        0b111: 0, 0b110: 0, 0b101: 0, 0b100: 1,
        0b011: 1, 0b010: 1, 0b001: 1, 0b000: 0
    }
    val = int(''.join(map(str, neighborhood)), 2)
    return rule_map[val]

def init_ca_grid(feature_vector, grid_size):
    feat_norm = (feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min() + 1e-8)
    grid = np.interp(
        np.linspace(0, len(feature_vector)-1, grid_size),
        np.arange(len(feature_vector)),
        feat_norm
    )
    grid = (grid > 0.5).astype(int)
    return grid

def ca_evolve(grid, steps, rule=rule30):
    grid_size = len(grid)
    history = [grid.copy()]
    current = grid.copy()

    for _ in range(steps):
        next_grid = np.zeros(grid_size, dtype=int)
        for i in range(grid_size):
            left = current[i-1] if i > 0 else 0
            mid = current[i]
            right = current[i+1] if i < grid_size-1 else 0
            next_grid[i] = rule([left, mid, right])
        current = next_grid
        history.append(current.copy())
    return np.array(history)

def extract_ca_features(ca_history):
    final_grid = ca_history[-1]
    density = np.mean(final_grid)

    counts = np.bincount(final_grid, minlength=2)
    ent = entropy(counts / (counts.sum() + 1e-8))

    max_conn = 0
    current = 0
    for v in final_grid:
        if v == 1:
            current += 1
            max_conn = max(max_conn, current)
        else:
            current = 0

    activity = np.sum(np.abs(np.diff(ca_history, axis=0))) / np.prod(ca_history.shape)
    global_mean = np.mean(ca_history)

    return np.array([density, ent, max_conn, activity, global_mean])

# -------------------------- 模型加载 --------------------------
def load_all(ca_params_path=None, nb_model_path=None):
    ca_path = ca_params_path or CA_PARAMS_PATH
    nb_path = nb_model_path or NB_MODEL_PATH
    try:
        logger.info("开始加载 CA 参数与朴素贝叶斯模型")

        with open(ca_path, "rb") as f:
            ca_params = pickle.load(f)
        with open(nb_path, "rb") as f:
            model_dict = pickle.load(f)

        nb_model = model_dict["nb_model"]
        scaler = model_dict["scaler"]

        logger.info("模型与参数加载完成")
        return ca_params, nb_model, scaler

    except Exception as e:
        logger.error("加载失败", exc_info=True)
        raise

# -------------------------- 推理主函数 --------------------------
def infer_ca_nb(infer_loader, has_label=True, ca_params_path=_SENTINEL, nb_model_path=_SENTINEL, infer_result_path=_SENTINEL):
    """
    :param ca_params_path: CA参数加载路径，不传则用默认
    :param nb_model_path: NB模型加载路径，不传则用默认
    :param infer_result_path: 推理结果保存路径，None 则不保存，不传则用默认
    """
    ca_path = CA_PARAMS_PATH if ca_params_path is _SENTINEL else ca_params_path
    nb_path = NB_MODEL_PATH if nb_model_path is _SENTINEL else nb_model_path
    res_path = INFER_RESULT_PATH if infer_result_path is _SENTINEL else infer_result_path
    try:
        ca_params, nb_model, scaler = load_all(ca_path, nb_path)
        grid_size = ca_params["grid_size"]
        evolve_steps = ca_params["evolve_steps"]

        logger.info("开始从 DataLoader 提取样本并提取 CA 特征")

        feats = []
        labels = []
        sample_ids = []

        for batch_idx, batch in enumerate(infer_loader):
            if has_label:
                data, lbl = batch
                lbl = lbl.cpu().numpy()
                labels.extend(lbl.tolist())
            else:
                data = batch

            data = data.cpu().numpy().reshape(data.shape[0], -1)
            bs = data.shape[0]

            for x in data:
                g = init_ca_grid(x, grid_size)
                h = ca_evolve(g, evolve_steps)
                f = extract_ca_features(h)
                feats.append(f)

            sample_ids.extend([f"b{batch_idx}_s{i}" for i in range(bs)])

        feats = np.array(feats)
        feats = scaler.transform(feats)

        logger.info(f"推理样本数: {len(sample_ids)}, CA 特征维度: {feats.shape[1]}")

        # 预测
        pred = nb_model.predict(feats)
        prob = nb_model.predict_proba(feats)
        max_prob = prob.max(axis=1)

        # 构造结果
        result = {
            "sample_count": len(sample_ids),
            "class_count": prob.shape[1],
            "predictions": [],
            "metrics": {}
        }

        for i, sid in enumerate(sample_ids):
            item = {
                "sample_id": sid,
                "pred": int(pred[i]),
                "confidence": float(max_prob[i])
            }
            if has_label:
                item["true_label"] = int(labels[i])
            result["predictions"].append(item)

        if has_label:
            acc = float(accuracy_score(labels, pred))
            result["metrics"]["accuracy"] = acc
            logger.info(f"推理准确率: {acc:.4f}")

        # 保存（路径为 None 则不保存）
        if res_path is not None:
            os.makedirs(os.path.dirname(res_path) or ".", exist_ok=True)
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f"推理结果已保存至 {res_path}")
        return result

    except Exception as e:
        logger.error("推理过程异常", exc_info=True)
        raise

# -------------------------- 测试入口 --------------------------
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    class MockTestSet(Dataset):
        def __len__(self):
            return 200
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0,10,()).item()

    loader = DataLoader(MockTestSet(), batch_size=32, shuffle=False)
    res = infer_ca_nb(loader, has_label=True)