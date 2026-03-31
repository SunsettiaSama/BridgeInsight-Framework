import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# 读取搜索结果
result_file = str(Path(__file__).parent.parent.parent.parent.parent) + '/results/training_result/deep_learning_module/search_best_hyperparams/mlp_search_result.json'

with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取每个参数组合的最佳F1值
results_with_f1 = []
for result in data['search_results']:
    if 'training_metadata' in result and 'epoch_states' in result['training_metadata']:
        epoch_states = result['training_metadata']['epoch_states']
        # 获取最高的验证F1值及其对应的其他指标
        best_epoch_idx = max(range(len(epoch_states)), key=lambda i: epoch_states[i]['val_metrics']['f1'])
        best_epoch = epoch_states[best_epoch_idx]
        best_val_metrics = best_epoch['val_metrics']
        
        best_val_f1 = best_val_metrics['f1']
        best_val_accuracy = best_val_metrics['accuracy']
        best_val_precision = best_val_metrics['precision']
        best_val_recall = best_val_metrics['recall']
        best_accuracy = result['accuracy']
        
        results_with_f1.append({
            'params': result['params'],
            'best_accuracy': best_accuracy,
            'best_val_f1': best_val_f1,
            'best_val_accuracy': best_val_accuracy,
            'best_val_precision': best_val_precision,
            'best_val_recall': best_val_recall,
            'epochs': result['epochs']
        })

# 按F1排序
sorted_by_f1 = sorted(results_with_f1, key=lambda x: x['best_val_f1'], reverse=True)

print('=' * 100)
print('【按F1值排序 - 所有参数组合结果详情】')
print('=' * 100)
for rank, result in enumerate(sorted_by_f1, 1):
    print(f'{rank}. F1={result["best_val_f1"]:.6f}, Acc={result["best_val_accuracy"]:.6f}, '
          f'Precision={result["best_val_precision"]:.6f}, Recall={result["best_val_recall"]:.6f}')
    print(f'   batch_size: {result["params"]["batch_size"]}, lr: {result["params"]["learning_rate"]}, '
          f'weight_decay: {result["params"]["weight_decay"]}, grad_clip: {result["params"]["gradient_clip_norm"]}')
    print()

# 按accuracy排序（原始排序方式）
sorted_by_acc = sorted(results_with_f1, key=lambda x: x['best_accuracy'], reverse=True)

print('=' * 100)
print('【按Accuracy排序 - 所有参数组合结果详情】')
print('=' * 100)
for rank, result in enumerate(sorted_by_acc, 1):
    print(f'{rank}. Acc={result["best_accuracy"]:.6f}, F1={result["best_val_f1"]:.6f}, '
          f'Precision={result["best_val_precision"]:.6f}, Recall={result["best_val_recall"]:.6f}')
    print(f'   batch_size: {result["params"]["batch_size"]}, lr: {result["params"]["learning_rate"]}, '
          f'weight_decay: {result["params"]["weight_decay"]}, grad_clip: {result["params"]["gradient_clip_norm"]}')
    print()
