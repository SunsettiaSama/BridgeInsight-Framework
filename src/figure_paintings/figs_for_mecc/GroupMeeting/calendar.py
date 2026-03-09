import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import calendar
from typing import Dict, Tuple, Optional, Union


font_size = 14

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = font_size


def plot_calendar_heatmap(
    year: int,
    month: int,
    day_colors: Dict[int, Union[str, Tuple[float, float, float]]],
    day_descriptions: Optional[Dict[int, str]] = None,
    legend_mapping: Optional[Dict[Union[str, Tuple[float, float, float]], str]] = None,
    annotation_text: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    default_color: str = "#f0f0f0",
    text_color: str = "#333333",
    grid_color: str = "#dddddd",
    weekday_labels: list = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"],
    weekday_label_padding: float = 0.3,
    desc_fontsize: int = 8,
    desc_color: str = "#666666",
    legend_fontsize: int = 9,
    legend_block_size: Tuple[float, float] = (0.3, 0.2),
    legend_spacing: float = 0.1,
    font_family: str = "SimHei"
) -> plt.Figure:
    """
    优化版日历热力图：
    - 解决右侧padding过多问题（动态计算画布范围）
    - 布局自适应，移除固定右侧留白
    - 保留原有所有功能
    """
    # ========== 输入合法性校验 ==========
    if not isinstance(year, int) or year < 1900 or year > 2100:
        raise ValueError("年份必须是1900-2100之间的整数")
    if not isinstance(month, int) or month < 1 or month > 12:
        raise ValueError("月份必须是1-12之间的整数")
    
    _, total_days = calendar.monthrange(year, month)
    invalid_days = [day for day in day_colors.keys() if day < 1 or day > total_days]
    if invalid_days:
        raise ValueError(f"{year}年{month}月无以下日期：{invalid_days}，该月总天数为{total_days}天")
    
    if day_descriptions is not None:
        if not isinstance(day_descriptions, dict):
            raise TypeError("day_descriptions必须是字典（key=日期，value=描述字符串）")
        invalid_desc_days = [day for day in day_descriptions.keys() if day < 1 or day > total_days]
        if invalid_desc_days:
            raise ValueError(f"{year}年{month}月无以下描述日期：{invalid_desc_days}")
    
    if legend_mapping is not None and not isinstance(legend_mapping, dict):
        raise TypeError("legend_mapping必须是字典（key=颜色值，value=标注文字）")

    # ========== 初始化画布 ==========
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize)
    
    # ========== 动态计算图例所需空间（核心优化1） ==========
    # 计算图例总宽度：色块宽度 + 文字间距 + 文字预估宽度（按字符数估算）
    legend_total_width = 0.0
    if legend_mapping and len(legend_mapping) > 0:
        # 预估单条图例文字宽度（每个字符约0.15单位）
        max_label_len = max(len(label) for label in legend_mapping.values()) if legend_mapping else 0
        legend_total_width = legend_block_size[0] + 0.1 + (max_label_len * 0.15)
    # 纯文字标注额外宽度
    if annotation_text:
        anno_width = len(annotation_text) * 0.15 + 0.3
        legend_total_width = max(legend_total_width, anno_width)
    
    # 动态设置X轴范围：日历宽度(7) + 图例宽度 + 少量边距
    x_max = 7.0 + legend_total_width + 0.2  # 仅保留必要边距
    ax.set_xlim(0, x_max)
    
    # 动态计算Y轴范围（核心优化2：移除冗余预留）
    legend_y_space = 0.0
    if legend_mapping and len(legend_mapping) > 0:
        legend_y_space = len(legend_mapping) * (legend_block_size[1] + legend_spacing)
    # 纯文字标注额外Y空间
    anno_y_space = 0.3 if annotation_text else 0.0
    y_max = 6 + weekday_label_padding + legend_y_space + anno_y_space
    ax.set_ylim(0, y_max)
    
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_frame_on(False)

    # ========== 生成日历网格数据 ==========
    cal_matrix = calendar.monthcalendar(year, month)
    while len(cal_matrix) < 6:
        cal_matrix.append([0]*7)

    # ========== 绘制日历热力块 + 日期 + 描述 ==========
    for week_idx, week in enumerate(cal_matrix):
        for day_idx, day in enumerate(week):
            x = day_idx + 0.5
            y = week_idx + 0.5
            
            rect = mpatches.Rectangle(
                (day_idx, week_idx), 1, 1,
                facecolor=day_colors.get(day, default_color) if day != 0 else "#ffffff",
                edgecolor=grid_color,
                linewidth=1
            )
            ax.add_patch(rect)
            
            if day != 0:
                ax.text(
                    x, y + 0.15, str(day),
                    ha="center", va="center",
                    color=text_color
                )
                if day_descriptions and day in day_descriptions:
                    desc = day_descriptions[day][:8] + "..." if len(day_descriptions[day]) > 8 else day_descriptions[day]
                    ax.text(
                        x, y - 0.2, desc,
                        ha="center", va="center",
                        color=desc_color, fontsize=desc_fontsize, style="italic"
                    )

    # ========== 绘制星期标签 ==========
    label_y = 6 + weekday_label_padding / 2
    for idx, label in enumerate(weekday_labels):
        ax.text(
            idx + 0.5, label_y, label,
            ha="center", va="center",
            color="#666666", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="none", alpha=0.5)
        )

    # ========== 绘制右上角图注（优化位置计算） ==========
    legend_start_x = 7.1  # 日历右侧紧邻位置（原7.0，微调避免重叠）
    legend_start_y = 6.0
    
    # 1. 绘制「色块+文字」图例
    if legend_mapping is not None and len(legend_mapping) > 0:
        current_y = legend_start_y
        for color, label in legend_mapping.items():
            legend_rect = mpatches.Rectangle(
                (legend_start_x, current_y - legend_block_size[1]/2),
                legend_block_size[0], legend_block_size[1],
                facecolor=color, edgecolor="#cccccc", linewidth=0.5
            )
            ax.add_patch(legend_rect)
            
            ax.text(
                legend_start_x + legend_block_size[0] + 0.1, current_y,
                label, ha="left", va="center",
                color="#222222", fontsize=legend_fontsize, weight="normal"
            )
            current_y -= (legend_block_size[1] + legend_spacing)
    
    # 2. 兼容纯文字标注
    if annotation_text is not None:
        anno_y = legend_start_y + 0.3 if legend_mapping else legend_start_y
        ax.text(
            legend_start_x, anno_y,
            annotation_text, ha="left", va="center",
            color="#222222", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.8)
        )

    # ========== 设置标题 ==========
    if title is None:
        title = f"{year}年{month:02d}月日历热力图"
    ax.text(
        3.5, -0.5, title, ha="center", va="center",
        color="#333333", weight="bold"
    )

    # ========== 调整布局（核心优化3：移除固定右侧留白） ==========
    plt.tight_layout()  # 自动适配布局
    # 仅保留必要的上下边距，移除固定的right/left强制设置
    plt.subplots_adjust(top=0.92, bottom=0.08)  
    return fig



import pandas as pd
import warnings
from typing import Optional, Dict, Tuple, Any
warnings.filterwarnings('ignore')


def analyze_daily_viv_from_parquet(
    parquet_path: str,
    timestamp_col: str = "timestamp",
    sensor_01_col: str = "ST-VIC-C34-102-01-data_isVIV",
    sensor_02_col: str = "ST-VIC-C34-102-02-data_isVIV",
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    从Parquet文件分析每日VIV发生状态（细粒度按日期统计），返回DataFrame和画图专用字典
    
    参数说明：
    ----------
    parquet_path : str
        Parquet文件的路径（本地路径）
    timestamp_col : str, optional
        时间戳列名，默认"timestamp"
    sensor_01_col : str, optional
        传感器1的VIV状态列名，默认"ST-VIC-C34-102-01-data_isVIV"
    sensor_02_col : str, optional
        传感器2的VIV状态列名，默认"ST-VIC-C34-102-02-data_isVIV"
    output_path : Optional[str], optional
        结果保存路径（CSV格式），None则不保存，默认None
    
    返回值：
    ----------
    Tuple[pd.DataFrame, Dict[int, int]]
        - DataFrame：每日VIV统计详情，含列：date（日期）、day（当月日数1-31）、viv_daily_flag（0/1/2/3）、
          sensor_01_occur（是否发生）、sensor_02_occur（是否发生）
        - Dict：画图专用字典，格式{日: 标记}，如{1:0, 4:1, 21:3}
    
    标记规则：
    ----------
    - 0：当日无任何VIV
    - 1：仅传感器ST-VIC-C34-102-01发生VIV
    - 2：仅传感器ST-VIC-C34-102-02发生VIV
    - 3：两个传感器都发生VIV
    """
    # ========== 1. 读取Parquet文件（异常处理） ==========
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Parquet文件不存在：{parquet_path}")
    except Exception as e:
        raise RuntimeError(f"读取Parquet文件失败：{str(e)}")

    # ========== 2. 验证必要列存在 ==========
    required_cols = [timestamp_col, sensor_01_col, sensor_02_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列：{missing_cols}")

    # ========== 3. 处理时间戳，提取日期和当月日数 ==========
    df["datetime"] = pd.to_datetime(df[timestamp_col])  # 转换为datetime
    df["date"] = df["datetime"].dt.date  # 完整日期（年-月-日）
    df["day"] = df["datetime"].dt.day    # 当月日数（1-31），用于字典key
    df_sorted = df.sort_values("datetime").reset_index(drop=True)

    # ========== 4. 按日期分组，计算当日VIV标记 ==========
    def _calc_daily_viv_flag(group: pd.DataFrame) -> Dict[str, Any]:
        """内部函数：计算单日期的VIV状态"""
        s1_happened = group[sensor_01_col].any()
        s2_happened = group[sensor_02_col].any()
        
        if not s1_happened and not s2_happened:
            flag = 0
        elif s1_happened and not s2_happened:
            flag = 1
        elif not s1_happened and s2_happened:
            flag = 2
        else:
            flag = 3
        
        return {
            "day": group["day"].iloc[0],  # 提取当日的"日"（1-31）
            "viv_daily_flag": flag,
            "sensor_01_occur": s1_happened,
            "sensor_02_occur": s2_happened
        }

    # 分组计算并整理结果
    daily_stats = df_sorted.groupby("date").apply(
        lambda g: pd.Series(_calc_daily_viv_flag(g)),
        include_groups=False
    ).reset_index()

    # 调整列顺序，提升可读性
    daily_stats = daily_stats[["date", "day", "viv_daily_flag", "sensor_01_occur", "sensor_02_occur"]]

    # ========== 5. 生成画图专用字典（{日: 标记}） ==========
    # 按日升序排列，确保字典key有序
    daily_stats_sorted = daily_stats.sort_values("day").reset_index(drop=True)
    viv_flag_dict = dict(zip(daily_stats_sorted["day"], daily_stats_sorted["viv_daily_flag"]))

    # ========== 6. 保存结果（可选） ==========
    if output_path:
        try:
            daily_stats.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"统计详情已保存至：{output_path}")
        except Exception as e:
            raise RuntimeError(f"保存结果失败：{str(e)}")

    # ========== 7. 返回结果 ==========
    return daily_stats, viv_flag_dict


def build_calculate_chunk():
    
    # 进行简单数据处理
    # from src.data_processer.persistence_utils import *
    # from src.data_processer.pipeline_orchestrator import *
    # from src.data_processer.algorithms import *

    from src.config import sensor_config
    from src.data_processer.database_manager import ChunkManager

    manager = ChunkManager(local_dir = r'F:\Research\Vibration Characteristics In Cable Vibration\data\202409db')

    df_blocks = manager.compute(
        target_col=[
                    # "ST-UAN-G04-001-01", 
                    'ST-VIC-C34-102-01', 
                    'ST-VIC-C34-102-02'
                    ],
        funcs=[
            # [np.mean, np.std],    # speed 列应用两个函数
            [isVIV],      # temperature 列应用两个函数
            [isVIV], 
        ], 
        n_workers = 4, 
        keep_original_data = True
        )

    result_blocks = df_blocks[60: ] # 筛除无用数据
    result_blocks = clean_by_length(result_blocks)
    result_blocks = clean_dataframe(result_blocks)
    result_blocks.to_parquet('./data/ST-VIC-C34-102-01_Inplane_Outplane_Cal.parquet')

    pass

# ------------------------------ 示例调用（完善日历绘图） ------------------------------
if __name__ == "__main__":

    # 替换为实际文件路径
    PARQUET_FILE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\data\ST-VIC-C34-102-01_Inplane_Outplane_Cal.parquet"

    # 执行分析
    try:
        stats_df, viv_flag_dict = analyze_daily_viv_from_parquet(
            parquet_path=PARQUET_FILE_PATH,
        )

        # 打印结果
        print("=" * 50)
        print("每日VIV统计详情（DataFrame）：")
        print(stats_df.head(10))

        print("\n" + "=" * 50)
        print("画图专用VIV标记字典：")
        print(viv_flag_dict)  # 格式示例：{1:0, 4:1, 21:1, 5:2, 6:3}


        # ------------------------------ 日历热力图绘制 ------------------------------
        # 1. 定义标记→颜色映射（对齐你提供的图中颜色）
        flag_color_map = {
            0: "#f0f0f0",    # 无VIV（默认色）
            1: "#257D8B",    # 面内（对应传感器1，图中深蓝色）
            2: "#68BED9",    # 面外（对应传感器2，图中浅蓝色）
            3: "#ED8D5A"     # 面内/面外（图中橙色）
        }

        # 2. 生成日历绘图所需的day_colors（日期→颜色）
        day_colors = {day: flag_color_map[flag] for day, flag in viv_flag_dict.items()}

        # 3. 定义图例映射（颜色→标签）
        legend_mapping = {
            flag_color_map[0]: " ",          # 无VIV（空标签）
            flag_color_map[1]: "面内",       # 仅传感器1发生
            flag_color_map[2]: "面外",       # 仅传感器2发生
            flag_color_map[3]: "面内/面外"   # 两者同时发生
        }

        # 4. 调用日历热力图函数（绘制9月）
        fig = plot_calendar_heatmap(
            year=2024,  # 需与Parquet文件中的年份一致，可根据实际数据调整
            month=9,
            day_colors=day_colors,
            legend_mapping=legend_mapping,
            figsize=(12, 7),
            desc_fontsize = font_size + 4, 
            legend_fontsize = font_size + 4, 
        )

        # 5. 显示/保存图片
        plt.show()
        # fig.savefig("202409_viv_calendar_heatmap.png", dpi=150, bbox_inches="tight")


    except Exception as e:
        print(f"执行出错：{str(e)}")

