import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from matplotlib.font_manager import FontProperties
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, ENG_FONT, CN_FONT, get_blue_color_map

CMAP = get_blue_color_map(style='gradient')
COLORS = CMAP(np.linspace(0, 1, 256))
CURVE_COLOR = COLORS[255]

BAYES_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\machine_learning_module\bayes\bayes_kfold_result.json"
Y_LIM = (0, 1)
    
def load_bayes_results(result_path):
    """加载贝叶斯超参数搜索结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def organize_results(search_results):
    """将搜索结果按单参数组织"""
    param_values = []
    f1_scores = []
    results_dict = {}
    
    for result in search_results:
        params = result['params']
        var_smoothing = params['var_smoothing']
        mean_f1 = result['mean_f1']
        
        param_values.append(var_smoothing)
        f1_scores.append(mean_f1)
        results_dict[var_smoothing] = result
    
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    f1_scores = np.array([f1_scores[i] for i in sorted_indices])
    
    return param_values, f1_scores, results_dict

def plot_bayes_f1_curve(param_values, f1_scores, results_dict, ploter):
    """绘制贝叶斯单参数与F1分数的曲线图"""
    
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    param_str_values = [f'{p:.0e}' if isinstance(p, float) else str(p) for p in param_values]
    x_positions = np.arange(len(param_values))
    
    ax.plot(x_positions, f1_scores, color=CURVE_COLOR, linewidth=3, marker='o', markersize=8)
    ax.fill_between(x_positions, f1_scores, alpha=0.3, color=CURVE_COLOR)
    
    best_idx = np.argmax(f1_scores)
    best_var_smoothing = param_values[best_idx]
    best_f1 = f1_scores[best_idx]
    best_result = results_dict[best_var_smoothing]
    
    print(f"\n{'=' * 80}")
    print(f"【Gaussian Naive Bayes - 最优参数】")
    print(f"{'=' * 80}")
    print(f"  var_smoothing = {best_var_smoothing}")
    print(f"  Mean Accuracy = {best_result['mean_accuracy']:.4f}")
    print(f"  Std Accuracy  = {best_result['std_accuracy']:.4f}")
    print(f"  Mean F1 Score = {best_result['mean_f1']:.4f}")
    print(f"{'=' * 80}\n")
    
    ax.set_xlabel('var_smoothing', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT)
    ax.set_title('Gaussian Naive Bayes Parameter Search', fontproperties=ENG_FONT, pad=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_str_values, rotation=45, ha='right', fontproperties=ENG_FONT)
    ax.set_ylim(Y_LIM)
    ax.grid(True, alpha=0.3)
    
    ax.tick_params(axis='y', labelsize=SQUARE_FONT_SIZE - 4)
    
    plt.tight_layout()
    
    ploter.figs.append(fig)

def main():
    """主函数"""
    print("=" * 80)
    print("开始绘制贝叶斯超参数搜索曲线图")
    print("=" * 80)
    
    if not os.path.exists(BAYES_RESULT_PATH):
        print(f"❌ 结果文件不存在：{BAYES_RESULT_PATH}")
        return
    
    data = load_bayes_results(BAYES_RESULT_PATH)
    search_results = data.get('kfold_search_results', [])
    
    if not search_results:
        print("❌ 搜索结果为空")
        return
    
    param_values, f1_scores, results_dict = organize_results(search_results)
    
    print(f"\n找到 {len(param_values)} 个参数配置")
    
    ploter = PlotLib()
    plot_bayes_f1_curve(param_values, f1_scores, results_dict, ploter)
    
    print(f"\n生成了 {len(ploter.figs)} 张曲线图")
    
    ploter.show()
    
    print("\n" + "=" * 80)
    print("所有曲线图绘制完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
