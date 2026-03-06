"""
标注数据集可视化工具

功能：
1. 展示标注结果的分布统计（各标签占比）
2. 展示传感器分布
3. 展示时间分布
4. 生成可视化图表
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import logging

# ==================== 常量定义 ====================

DEFAULT_JSON_RESULTS_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\dataset_annotation\annotation_results.json"

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# matplotlib 配置
MATPLOTLIB_FONT = 'SimHei'
MATPLOTLIB_ENABLE_UNICODE_MINUS = False

# 图表配置
PIE_CHART_FIGSIZE = (10, 6)
PIE_CHART_COLORS = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
PIE_CHART_AUTOPCT_FORMAT = '%1.1f%%'
PIE_CHART_TITLE_FONTSIZE = 14
PIE_CHART_LABEL_FONTSIZE = 12
PIE_CHART_AUTOTEXT_FONTSIZE = 11
PIE_CHART_START_ANGLE = 90

BAR_CHART_FIGSIZE = (12, 6)
BAR_CHART_COLOR = 'steelblue'
BAR_CHART_ALPHA = 0.8
BAR_CHART_TITLE_FONTSIZE = 14
BAR_CHART_LABEL_FONTSIZE = 12
BAR_CHART_AUTOTEXT_DPI = 300

# 文本输出配置
SUMMARY_LINE_LENGTH = 70
PROGRESS_BAR_LENGTH = 50
DETAILED_REPORT_PREVIEW_COUNT = 5
DETAILED_REPORT_PATH_PREVIEW_LENGTH = 30

# ==================== 日志配置 ====================

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


class AnnotationDatasetVisualizer:
    """标注数据集可视化工具"""
    
    def __init__(self, json_results_path: str = None):
        """
        初始化可视化工具
        
        Args:
            json_results_path: JSON标注结果文件路径，默认使用DEFAULT_JSON_RESULTS_PATH
        """
        self.json_path = Path(json_results_path or DEFAULT_JSON_RESULTS_PATH)
        self.results = []
        self.stats = {}
        
        # 加载数据
        self._load_results()
    
    def _load_results(self) -> None:
        """加载JSON标注结果"""
        if not self.json_path.exists():
            logger.warning(f"文件不存在：{self.json_path}")
            self.results = []
            return
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            logger.info(f"[OK] 加载{len(self.results)}条标注记录")
        except Exception as e:
            logger.error(f"加载JSON失败：{e}")
            self.results = []
    
    # =====================
    # 1. 标注结果分布统计
    # =====================
    
    def get_annotation_distribution(self) -> Dict[str, int]:
        """
        获取标注结果的分布
        
        Returns:
            标注值和对应的数量，如 {'0': 100, '1': 50, '2': 30}
        """
        if not self.results:
            logger.warning("没有标注数据")
            return {}
        
        annotations = [item.get('annotation', '').strip() for item in self.results]
        distribution = dict(Counter(annotations))
        
        return dict(sorted(distribution.items()))
    
    def get_annotation_percentage(self) -> Dict[str, float]:
        """
        获取标注结果的百分比分布
        
        Returns:
            标注值和对应的百分比，如 {'0': 50.0, '1': 25.0, '2': 15.0}
        """
        distribution = self.get_annotation_distribution()
        total = sum(distribution.values())
        
        if total == 0:
            return {}
        
        percentage = {
            label: (count / total) * 100
            for label, count in distribution.items()
        }
        
        return percentage
    
    # =====================
    # 2. 传感器分布统计
    # =====================
    
    def get_sensor_distribution(self) -> Dict[str, int]:
        """
        获取各传感器的标注数量分布
        
        Returns:
            传感器ID和对应的标注数量
        """
        if not self.results:
            return {}
        
        sensors = [item.get('sensor_id', 'unknown') for item in self.results]
        distribution = dict(Counter(sensors))
        
        return dict(sorted(distribution.items()))
    
    def get_sensor_annotation_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        获取各传感器的标注结果分布
        
        Returns:
            {传感器ID: {标注值: 数量}}
        """
        if not self.results:
            return {}
        
        sensor_annotations = {}
        
        for item in self.results:
            sensor_id = item.get('sensor_id', 'unknown')
            annotation = item.get('annotation', '').strip()
            
            if sensor_id not in sensor_annotations:
                sensor_annotations[sensor_id] = Counter()
            
            sensor_annotations[sensor_id][annotation] += 1
        
        # 转为普通字典
        result = {
            sensor: dict(counts)
            for sensor, counts in sensor_annotations.items()
        }
        
        return dict(sorted(result.items()))
    
    # =====================
    # 3. 文件分布统计
    # =====================
    
    def get_file_distribution(self) -> Dict[str, int]:
        """
        获取各文件的标注数量分布
        
        Returns:
            文件路径和对应的标注数量
        """
        if not self.results:
            return {}
        
        files = [item.get('file_path', 'unknown') for item in self.results]
        distribution = dict(Counter(files))
        
        return dict(sorted(distribution.items()))
    
    # =====================
    # 4. 综合统计
    # =====================
    
    def compute_stats(self) -> Dict[str, Any]:
        """
        计算所有统计信息
        
        Returns:
            包含所有统计信息的字典
        """
        self.stats = {
            'total_records': len(self.results),
            'annotation_distribution': self.get_annotation_distribution(),
            'annotation_percentage': self.get_annotation_percentage(),
            'sensor_distribution': self.get_sensor_distribution(),
            'sensor_annotation_distribution': self.get_sensor_annotation_distribution(),
            'file_count': len(self.get_file_distribution()),
            'unique_sensors': len(self.get_sensor_distribution()),
        }
        return self.stats
    
    # =====================
    # 5. 文本输出展示
    # =====================
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        self.compute_stats()
        
        print("\n" + "="*SUMMARY_LINE_LENGTH)
        print("标注数据集统计摘要".center(SUMMARY_LINE_LENGTH))
        print("="*SUMMARY_LINE_LENGTH)
        
        # 1. 总体信息
        print(f"\n【基本信息】")
        print(f"  总标注记录数: {self.stats['total_records']}")
        print(f"  涉及文件数: {self.stats['file_count']}")
        print(f"  涉及传感器数: {self.stats['unique_sensors']}")
        
        # 2. 标注结果分布
        print(f"\n【标注结果分布】")
        distribution = self.stats['annotation_distribution']
        percentage = self.stats['annotation_percentage']
        
        if not distribution:
            print("  无标注数据")
        else:
            for label in sorted(distribution.keys()):
                count = distribution[label]
                pct = percentage.get(label, 0)
                bar_length = int(pct / 2)  # 使用PROGRESS_BAR_LENGTH个字符为满
                bar = "█" * bar_length + "░" * (PROGRESS_BAR_LENGTH - bar_length)
                print(f"  标注 {label:>3}: {count:>4} 个  ({pct:>6.2f}%) {bar}")
        
        # 3. 传感器分布
        print(f"\n【传感器标注分布】")
        sensor_dist = self.stats['sensor_distribution']
        
        if not sensor_dist:
            print("  无传感器数据")
        else:
            for sensor_id, count in sensor_dist.items():
                pct = (count / self.stats['total_records']) * 100
                print(f"  {sensor_id}: {count:>4} 个 ({pct:>6.2f}%)")
        
        # 4. 各传感器的标注分布
        print(f"\n【各传感器的标注细分】")
        sensor_anno_dist = self.stats['sensor_annotation_distribution']
        
        if not sensor_anno_dist:
            print("  无数据")
        else:
            for sensor_id, anno_counts in sensor_anno_dist.items():
                print(f"\n  {sensor_id}:")
                for label in sorted(anno_counts.keys()):
                    count = anno_counts[label]
                    pct = (count / sensor_dist[sensor_id]) * 100
                    print(f"    标注 {label:>3}: {count:>3} 个 ({pct:>6.2f}%)")
        
        print("\n" + "="*SUMMARY_LINE_LENGTH + "\n")
    
    def print_detailed_report(self) -> None:
        """打印详细报告"""
        self.compute_stats()
        
        print("\n" + "="*SUMMARY_LINE_LENGTH)
        print("标注数据集详细报告".center(SUMMARY_LINE_LENGTH))
        print("="*SUMMARY_LINE_LENGTH)
        
        # 按标注值分类列表
        print(f"\n【按标注值分类的文件列表】")
        
        # 按标注值组织
        by_annotation = {}
        for item in self.results:
            anno = item.get('annotation', 'unknown').strip()
            if anno not in by_annotation:
                by_annotation[anno] = []
            by_annotation[anno].append(item)
        
        # 打印每个标注类别
        for label in sorted(by_annotation.keys()):
            items = by_annotation[label]
            print(f"\n  标注值: {label} (共 {len(items)} 条)")
            print(f"  {'-'*(SUMMARY_LINE_LENGTH - 4)}")
            
            for item in items[:DETAILED_REPORT_PREVIEW_COUNT]:
                file_path = item.get('file_path', 'unknown')
                window_idx = item.get('window_index', 'N/A')
                sensor_id = item.get('sensor_id', 'unknown')
                print(f"    窗口: {window_idx:>4} | 传感器: {sensor_id:<20} | 路径: ...{file_path[-DETAILED_REPORT_PATH_PREVIEW_LENGTH:]}")
            
            if len(items) > DETAILED_REPORT_PREVIEW_COUNT:
                print(f"    ... 还有 {len(items) - DETAILED_REPORT_PREVIEW_COUNT} 条记录未显示")
        
        print("\n" + "="*SUMMARY_LINE_LENGTH + "\n")
    
    # =====================
    # 6. 图表生成（可视化）
    # =====================
    
    def plot_annotation_distribution(self, save_path: Optional[str] = None) -> None:
        """
        绘制标注分布饼图
        
        Args:
            save_path: 可选的保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT]
            matplotlib.rcParams['axes.unicode_minus'] = MATPLOTLIB_ENABLE_UNICODE_MINUS
        except ImportError:
            logger.warning("matplotlib未安装，跳过图表生成")
            return
        
        distribution = self.get_annotation_distribution()
        
        if not distribution:
            logger.warning("没有数据可绘制")
            return
        
        # 创建饼图
        fig, ax = plt.subplots(figsize=PIE_CHART_FIGSIZE)
        
        labels = list(distribution.keys())
        sizes = list(distribution.values())
        colors = PIE_CHART_COLORS
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=PIE_CHART_AUTOPCT_FORMAT,
            colors=colors[:len(labels)],
            startangle=PIE_CHART_START_ANGLE
        )
        
        # 美化文字
        for text in texts:
            text.set_fontsize(PIE_CHART_LABEL_FONTSIZE)
            text.set_weight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(PIE_CHART_AUTOTEXT_FONTSIZE)
            autotext.set_weight('bold')
        
        ax.set_title('VIV标注结果分布', fontsize=PIE_CHART_TITLE_FONTSIZE, fontweight='bold', pad=20)
        
        # 添加图例
        total = sum(sizes)
        legend_labels = [f'标注 {label}: {sizes[i]} 个' for i, label in enumerate(labels)]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=BAR_CHART_AUTOTEXT_DPI, bbox_inches='tight')
            logger.info(f"✓ 图表已保存：{save_path}")
        else:
            plt.show()
    
    def plot_sensor_distribution(self, save_path: Optional[str] = None) -> None:
        """
        绘制传感器分布柱状图
        
        Args:
            save_path: 可选的保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT]
            matplotlib.rcParams['axes.unicode_minus'] = MATPLOTLIB_ENABLE_UNICODE_MINUS
        except ImportError:
            logger.warning("matplotlib未安装，跳过图表生成")
            return
        
        distribution = self.get_sensor_distribution()
        
        if not distribution:
            logger.warning("没有传感器数据")
            return
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=BAR_CHART_FIGSIZE)
        
        sensors = list(distribution.keys())
        counts = list(distribution.values())
        
        bars = ax.bar(range(len(sensors)), counts, color=BAR_CHART_COLOR, alpha=BAR_CHART_ALPHA)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold'
            )
        
        ax.set_xlabel('传感器ID', fontsize=BAR_CHART_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('标注数量', fontsize=BAR_CHART_LABEL_FONTSIZE, fontweight='bold')
        ax.set_title('各传感器标注分布', fontsize=BAR_CHART_TITLE_FONTSIZE, fontweight='bold', pad=20)
        ax.set_xticks(range(len(sensors)))
        ax.set_xticklabels(sensors, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=BAR_CHART_AUTOTEXT_DPI, bbox_inches='tight')
            logger.info(f"✓ 图表已保存：{save_path}")
        else:
            plt.show()


def main():
    """主函数 - 生成完整的统计报告"""
    try:
        visualizer = AnnotationDatasetVisualizer()
        
        # 打印摘要
        visualizer.print_summary()
        
        # 打印详细报告
        visualizer.print_detailed_report()
        
        # 生成图表（可选）
        # output_dir = Path("./output_charts")
        # output_dir.mkdir(exist_ok=True)
        # visualizer.plot_annotation_distribution(str(output_dir / "annotation_distribution.png"))
        # visualizer.plot_sensor_distribution(str(output_dir / "sensor_distribution.png"))
        
        return 0
    except Exception as e:
        logger.error(f"执行失败：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
