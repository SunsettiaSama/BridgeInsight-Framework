import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


CA_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\machine_learning_module\cellunar_automata\ca_result.json"


def load_ca_results(result_path):
    """加载元胞自动机超参数搜索结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def print_markdown_table_header(columns, col_widths):
    """打印markdown表格头"""
    header_row = "| " + " | ".join(f"{col:^{width}}" for col, width in zip(columns, col_widths)) + " |"
    separator_row = "|" + "|".join(f"-{'-' * (width)}-" for width in col_widths) + "|"
    print(header_row)
    print(separator_row)


def print_markdown_table_row(values, col_widths):
    """打印markdown表格行"""
    row = "| " + " | ".join(f"{str(v):^{width}}" for v, width in zip(values, col_widths)) + " |"
    print(row)


def print_ca_search_results_table(search_results):
    """以表格形式打印元胞自动机搜索结果"""
    
    if not search_results:
        print("❌ 搜索结果为空")
        return
    
    print("\n" + "=" * 150)
    print("【元胞自动机超参数搜索结果 - 完整表格】")
    print("=" * 150)
    
    columns = [
        "序号",
        "ca_grid",
        "ca_steps",
        "neighborhood_radius",
        "boundary_condition",
        "update_rule",
        "准确率",
        "精确率",
        "召回率",
        "F1分数"
    ]
    
    col_widths = [4, 8, 8, 18, 18, 14, 8, 8, 8, 8]
    
    print_markdown_table_header(columns, col_widths)
    
    for idx, result in enumerate(search_results, 1):
        params = result['params']
        
        values = [
            idx,
            params.get('ca_grid', '-'),
            params.get('ca_steps', '-'),
            params.get('neighborhood_radius', '-'),
            params.get('boundary_condition', '-'),
            params.get('update_rule', '-'),
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}"
        ]
        
        print_markdown_table_row(values, col_widths)
    
    print("=" * 150 + "\n")


def print_best_parameters_summary(best_params, search_results):
    """打印最优参数总结"""
    
    if not search_results:
        return
    
    best_acc_result = max(search_results, key=lambda x: x['accuracy'])
    best_f1_result = max(search_results, key=lambda x: x['f1'])
    
    print("\n" + "=" * 100)
    print("【最优参数配置总结】")
    print("=" * 100)
    
    print("\n📌 基于准确率的最优参数：")
    print("-" * 100)
    params_acc = best_acc_result['params']
    print(f"  • ca_grid:              {params_acc.get('ca_grid', '-')}")
    print(f"  • ca_steps:             {params_acc.get('ca_steps', '-')}")
    print(f"  • neighborhood_radius:  {params_acc.get('neighborhood_radius', '-')}")
    print(f"  • boundary_condition:   {params_acc.get('boundary_condition', '-')}")
    print(f"  • update_rule:          {params_acc.get('update_rule', '-')}")
    print(f"\n  性能指标：")
    print(f"    - 准确率: {best_acc_result['accuracy']:.4f}")
    print(f"    - 精确率: {best_acc_result['precision']:.4f}")
    print(f"    - 召回率: {best_acc_result['recall']:.4f}")
    print(f"    - F1分数: {best_acc_result['f1']:.4f}")
    
    print("\n📌 基于F1分数的最优参数：")
    print("-" * 100)
    params_f1 = best_f1_result['params']
    print(f"  • ca_grid:              {params_f1.get('ca_grid', '-')}")
    print(f"  • ca_steps:             {params_f1.get('ca_steps', '-')}")
    print(f"  • neighborhood_radius:  {params_f1.get('neighborhood_radius', '-')}")
    print(f"  • boundary_condition:   {params_f1.get('boundary_condition', '-')}")
    print(f"  • update_rule:          {params_f1.get('update_rule', '-')}")
    print(f"\n  性能指标：")
    print(f"    - 准确率: {best_f1_result['accuracy']:.4f}")
    print(f"    - 精确率: {best_f1_result['precision']:.4f}")
    print(f"    - 召回率: {best_f1_result['recall']:.4f}")
    print(f"    - F1分数: {best_f1_result['f1']:.4f}")
    
    print("\n" + "=" * 100 + "\n")


def print_statistics_summary(search_results):
    """打印统计信息总结"""
    
    if not search_results:
        return
    
    accuracies = [r['accuracy'] for r in search_results]
    f1_scores = [r['f1'] for r in search_results]
    
    print("\n" + "=" * 100)
    print("【搜索结果统计信息】")
    print("=" * 100)
    
    stats_columns = [
        "指标",
        "最小值",
        "最大值",
        "平均值",
        "中位数"
    ]
    stats_widths = [15, 12, 12, 12, 12]
    
    print_markdown_table_header(stats_columns, stats_widths)
    
    acc_stats = [
        "准确率",
        f"{min(accuracies):.4f}",
        f"{max(accuracies):.4f}",
        f"{sum(accuracies) / len(accuracies):.4f}",
        f"{sorted(accuracies)[len(accuracies)//2]:.4f}"
    ]
    print_markdown_table_row(acc_stats, stats_widths)
    
    f1_stats = [
        "F1分数",
        f"{min(f1_scores):.4f}",
        f"{max(f1_scores):.4f}",
        f"{sum(f1_scores) / len(f1_scores):.4f}",
        f"{sorted(f1_scores)[len(f1_scores)//2]:.4f}"
    ]
    print_markdown_table_row(f1_stats, stats_widths)
    
    print("=" * 100 + "\n")


def print_parameter_analysis(search_results):
    """按各参数维度分析性能"""
    
    if not search_results:
        return
    
    print("\n" + "=" * 100)
    print("【参数维度分析 - 各参数平均性能】")
    print("=" * 100)
    
    param_analysis = {
        'ca_grid': {},
        'ca_steps': {},
        'neighborhood_radius': {},
        'boundary_condition': {},
        'update_rule': {}
    }
    
    for result in search_results:
        params = result['params']
        for key in param_analysis.keys():
            param_value = params.get(key)
            if param_value not in param_analysis[key]:
                param_analysis[key][param_value] = {'accuracies': [], 'f1s': []}
            param_analysis[key][param_value]['accuracies'].append(result['accuracy'])
            param_analysis[key][param_value]['f1s'].append(result['f1'])
    
    for param_name, param_values in param_analysis.items():
        print(f"\n📊 参数：{param_name}")
        print("-" * 100)
        
        analysis_columns = ["参数值", "平均准确率", "平均F1分数", "出现次数"]
        analysis_widths = [20, 15, 15, 12]
        print_markdown_table_header(analysis_columns, analysis_widths)
        
        for value in sorted(param_values.keys(), key=str):
            data = param_values[value]
            avg_acc = sum(data['accuracies']) / len(data['accuracies'])
            avg_f1 = sum(data['f1s']) / len(data['f1s'])
            count = len(data['accuracies'])
            
            row_values = [
                str(value),
                f"{avg_acc:.4f}",
                f"{avg_f1:.4f}",
                count
            ]
            print_markdown_table_row(row_values, analysis_widths)
    
    print("\n" + "=" * 100 + "\n")


def main():
    """主函数"""
    print("\n" + "=" * 100)
    print("元胞自动机超参数搜索结果展示")
    print("=" * 100)
    
    if not os.path.exists(CA_RESULT_PATH):
        print(f"❌ 结果文件不存在：{CA_RESULT_PATH}")
        return
    
    data = load_ca_results(CA_RESULT_PATH)
    search_results = data.get('search_results', [])
    best_params = data.get('best_params', {})
    
    if not search_results:
        print("❌ 搜索结果为空")
        return
    
    print(f"\n✅ 成功加载 {len(search_results)} 个参数组合的搜索结果\n")
    
    print_ca_search_results_table(search_results)
    
    print_best_parameters_summary(best_params, search_results)
    
    print_statistics_summary(search_results)
    
    print_parameter_analysis(search_results)
    
    print("\n" + "=" * 100)
    print("✅ 元胞自动机搜索结果展示完成！")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
