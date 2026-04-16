import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
result_dir   = project_root / "results" / "identification_result"

result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
if not result_files:
    raise FileNotFoundError(f"未找到识别结果文件，目录：{result_dir}")

result_path = result_files[-1]
print(f"结果文件：{result_path.name}\n")

with open(result_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

meta = payload.get("metadata", {})
print("=" * 60)
print("文件元信息")
print("=" * 60)
for k, v in meta.items():
    print(f"  {k}: {v}")

CLASS_NAMES = {0: "随机振动", 1: "涡激共振", 2: "风雨振", 3: "其他振动"}

predictions     = payload["predictions"]
sample_metadata = payload.get("sample_metadata", {})

print("\n" + "=" * 60)
print("前 10 条识别结果")
print("=" * 60)

for i, (idx_str, label) in enumerate(list(predictions.items())[:10]):
    sm    = sample_metadata.get(idx_str, {})
    sid   = sm.get("inplane_sensor_id", "-")
    ts    = sm.get("timestamp", [])
    win   = sm.get("window_idx", "-")
    label_name = CLASS_NAMES.get(label, f"未知({label})")
    print(
        f"  [{i:02d}] idx={idx_str:>7s}  传感器={sid:<24s}  "
        f"时间={ts}  窗口={win:>4}  预测={label}({label_name})"
    )

print("\n" + "=" * 60)
print("预测类别分布（全量）")
print("=" * 60)
from collections import Counter
dist = Counter(predictions.values())
total = len(predictions)
for label in sorted(dist):
    cnt = dist[label]
    print(f"  {label}({CLASS_NAMES.get(label, '?')}): {cnt:>8,d}  ({100*cnt/total:5.2f}%)")
print(f"  总计: {total:,d}")
