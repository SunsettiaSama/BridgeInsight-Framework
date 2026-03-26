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

SVM_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\machine_learning_module\svm\svm_kfold_result.json"

def load_svm_results(result_path):
    """加载SVM超参数搜索结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def organize_results_by_kernel(search_results):
    """将搜索结果按kernel类型组织"""
    kernel_dict = {}
    
    for result in search_results:
        params = result['params']
        kernel = params['kernel']
        C = params['C']
        gamma = params['gamma']
        mean_f1 = result['mean_f1']
        
        if kernel not in kernel_dict:
            kernel_dict[kernel] = {
                'C_values': [],
                'gamma_values': [],
                'f1_matrix': {}
            }
        
        if C not in kernel_dict[kernel]['C_values']:
            kernel_dict[kernel]['C_values'].append(C)
        if gamma not in kernel_dict[kernel]['gamma_values']:
            kernel_dict[kernel]['gamma_values'].append(gamma)
        
        kernel_dict[kernel]['f1_matrix'][(C, gamma)] = mean_f1
    
    for kernel in kernel_dict:
        kernel_dict[kernel]['C_values'] = sorted(kernel_dict[kernel]['C_values'])
        gamma_numeric = []
        gamma_str = []
        for g in kernel_dict[kernel]['gamma_values']:
            if isinstance(g, str):
                gamma_str.append(g)
            else:
                gamma_numeric.append(g)
        gamma_numeric.sort()
        kernel_dict[kernel]['gamma_values'] = gamma_numeric + gamma_str
    
    return kernel_dict

def plot_svm_contour_by_kernel(kernel_dict, ploter):
    """为每个kernel绘制C和gamma的等高线图（基于F1分数）"""
    
    cmap = get_blue_color_map(style='gradient')
    
    for kernel, data in kernel_dict.items():
        C_values = np.array(data['C_values'])
        gamma_values = data['gamma_values']
        f1_matrix_dict = data['f1_matrix']
        
        f1_matrix = np.zeros((len(gamma_values), len(C_values)))
        
        for i, gamma in enumerate(gamma_values):
            for j, C in enumerate(C_values):
                f1_matrix[i, j] = f1_matrix_dict.get((C, gamma), np.nan)
        
        fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
        X, Y = np.meshgrid(C_values, np.arange(len(gamma_values)))
        
        contour_plot = ax.contourf(
            X, Y, f1_matrix,
            cmap=cmap,
            levels=20,
            alpha=1.0
        )
        ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)
        
        best_idx = np.nanargmax(f1_matrix)
        best_i, best_j = np.unravel_index(best_idx, f1_matrix.shape)
        best_C = C_values[best_j]
        best_gamma = gamma_values[best_i]
        best_f1 = f1_matrix[best_i, best_j]
        
        gamma_str = str(best_gamma) if isinstance(best_gamma, str) else f'{best_gamma}'
        print(f"\n【{kernel.upper()} Kernel - 最优参数】")
        print(f"  C = {best_C}")
        print(f"  γ = {gamma_str}")
        print(f"  Mean F1 Score = {best_f1:.4f}")
        
        ax.set_xlabel('C', labelpad=10, fontproperties=ENG_FONT)
        ax.set_ylabel(r'$\gamma$', labelpad=10, fontproperties=ENG_FONT)
        ax.set_title(f'SVM Parameter Search: {kernel.upper()} Kernel', fontproperties=ENG_FONT, pad=20)
        
        ax.set_xscale('log')
        
        gamma_labels = [str(g) for g in gamma_values]
        ax.set_yticks(np.arange(len(gamma_values)))
        ax.set_yticklabels(gamma_labels, fontproperties=ENG_FONT)
        
        ax.tick_params(axis='x', labelsize=SQUARE_FONT_SIZE - 4)
        ax.tick_params(axis='y', labelsize=SQUARE_FONT_SIZE - 4)
        
        cbar = plt.colorbar(contour_plot, ax=ax, shrink=0.8)
        cbar.set_label('Mean F1 Score', fontproperties=ENG_FONT)
        
        plt.tight_layout()
        
        ploter.figs.append(fig)

def main():
    """主函数"""
    print("=" * 80)
    print("开始绘制SVM超参数搜索等高线图")
    print("=" * 80)
    
    if not os.path.exists(SVM_RESULT_PATH):
        print(f"❌ 结果文件不存在：{SVM_RESULT_PATH}")
        return
    
    data = load_svm_results(SVM_RESULT_PATH)
    search_results = data.get('kfold_search_results', [])
    
    if not search_results:
        print("❌ 搜索结果为空")
        return
    
    kernel_dict = organize_results_by_kernel(search_results)
    
    print(f"\n找到 {len(kernel_dict)} 个kernel类型：{list(kernel_dict.keys())}")
    
    ploter = PlotLib()
    plot_svm_contour_by_kernel(kernel_dict, ploter)
    
    print(f"\n生成了 {len(ploter.figs)} 张等高线图")
    
    ploter.show()
    
    print("\n" + "=" * 80)
    print("所有等高线图绘制完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
