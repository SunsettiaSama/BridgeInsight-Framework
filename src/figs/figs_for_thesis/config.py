"""
绘图基础配置文件
从fig2_2_time_series_rms.py中提取的配置，作为后续绘图的基础配置
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ==================== 字号配置 ====================
FONT_SIZE = 16
LABEL_FONT_SIZE = 20
FIG_SIZE = (12, 8)


# ==================== 字体配置 ====================
# 1. 全局默认字体设为Times New Roman（优先渲染英文，无中文乱码风险）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
# 2. 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FONT_SIZE

# 3. 定义中英文字体对象
ENG_FONT = FontProperties(family='Times New Roman', size=FONT_SIZE)
CN_FONT = FontProperties(
    family='SimSun', 
    size=FONT_SIZE,
    # Windows系统无需指定fname，Linux/Mac需手动填写SimHei字体文件路径，示例：
    # fname='/usr/share/fonts/truetype/simhei/SimHei.ttf'
)

# ==================== matplotlib样式配置 ====================
plt.style.use('default')

# ==================== 颜色映射配置 ====================
# 随机振动（一般振动）颜色
NORMAL_VIB_COLOR = '#D0D0D0'
NORMAL_EDGE_COLOR = '#A0A0A0'

# 涡激共振（VIV）颜色
VIV_VIB_COLOR = '#606060'
VIV_EDGE_COLOR = '#303030'

# 阈值线颜色
THRESHOLD_COLOR = '#202020'   # 深灰色

# VIV样本透明度（区分于随机振动的不透明）
VIV_ALPHA = 0.6

# ==================== 其他绘图配置 ====================
# 高粒度分箱数
N_BINS = 100


# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0   # 计算RMS的时间窗口（秒）
NFFT = 512
# RMS_TRHESHOLD = 0.16 # RMS阈值已改为动态计算（95%分位值）

# 颜色配置（根据阈值区分）
BELOW_THRESHOLD_COLOR = '#8074C8'  # 小于标准差阈值的颜色
ABOVE_THRESHOLD_COLOR = '#7895C1'  # 大于标准差阈值的颜色
DEFAULT_COLOR = '#8074C8'

#　配色ＣＭＡＰ
def get_color_map(style='discrete'):
    """
    获取图中树状图的配色映射
    :param style: 色图模式，可选 'discrete'（离散色图）或 'gradient'（渐变插值色图）
    :return: matplotlib.colors.Colormap 对象
    """
    # 从图中提取的配色方案（按环形图颜色顺序排列）
    color_hex = [
        '#8074C8',  # Bsal/M36 family 1
        '#7895C1',  # Bsal/M36 family 2
        '#A8CBDF',  # Bsal/M36 family 3
        '#D6EFF4',  # Bsal/M36 family 4
        '#F2FAFC',  # Bsal/M36 family 5
        '#F7FBC9',  # Bsal/M36 family 6
        '#F5EBAE',  # Singleton outliers
        '#F0C284',  # Eh/M36 family
        '#EF8B67',  # Bsal, Bd, Eh and Hp M36 family
        '#E3625D',  # Bsal and Bd M36 family
        '#B54764',  # Bsal/M36 family (dark red)
        '#992224'   # Bsal/M36 family (darkest red)
    ]
    
    if style == 'discrete':
        # 离散色图，直接使用原始颜色
        cmap = ListedColormap(color_hex, name='tree_family_cmap')
    elif style == 'gradient':
        # 渐变色图，对原始颜色进行平滑插值
        cmap = LinearSegmentedColormap.from_list('tree_family_gradient', color_hex, N=256)
    else:
        raise ValueError("style 参数仅支持 'discrete' 或 'gradient'")
    
    return cmap


# ------------------------------
# 测试代码（可选）
# ------------------------------
if __name__ == "__main__":
    # 获取两种模式的色图
    cmap_discrete = get_color_map(style='discrete')
    cmap_gradient = get_color_map(style='gradient')
    
    # 可视化色图效果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3))
    
    # 离散色图展示
    ax1.imshow([range(12)], cmap=cmap_discrete, aspect='auto')
    ax1.set_title("Discrete Color Map (12 Categories)")
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([f'Cat {i+1}' for i in range(12)], rotation=45, ha='right')
    ax1.set_yticks([])
    
    # 渐变色图展示
    ax2.imshow([range(256)], cmap=cmap_gradient, aspect='auto')
    ax2.set_title("Gradient Color Map (256 Interpolated Colors)")
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()
