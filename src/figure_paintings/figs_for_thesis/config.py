"""
绘图基础配置文件
从fig2_2_time_series_rms.py中提取的配置，作为后续绘图的基础配置
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from src.config.data_processer.preprocess.vibration_io_process.config import *



# ==================== 字号配置 ====================
FONT_SIZE = 20
LABEL_FONT_SIZE = 20
REC_FIG_SIZE = (12, 8) # 矩形配置，Word图像大小为12 cm
SQUARE_FIG_SIZE = (10, 8) # 正方形配置，Word图像大小为7.5cm

REC_FONT_SIZE = 22
SQUARE_FONT_SIZE = 24



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

# 颜色配置（根据阈值区分）
BELOW_THRESHOLD_COLOR = '#8074C8'  # 小于标准差阈值的颜色
ABOVE_THRESHOLD_COLOR = '#7895C1'  # 大于标准差阈值的颜色
DEFAULT_COLOR = '#8074C8'
ANNOTATION_COLOR = '#404040'


#　配色ＣＭＡＰ
def get_full_color_map(style='discrete'):
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

def get_blue_color_map(style='discrete', start_map_index = 0, end_map_index = 4):
    """
    获取蓝色系配色映射（对应原图中 Bsal/M36 family 1-5）
    :param style: 色图模式，可选 'discrete'（离散色图）或 'gradient'（渐变插值色图）
    :return: matplotlib.colors.Colormap 对象
    """
    # 按“浅→深”顺序排列（数值低→浅，数值高→深）
    blue_hex = [
        '#F2FAFC',  # Bsal/M36 family 5 (最浅)
        '#D6EFF4',  # Bsal/M36 family 4
        '#A8CBDF',  # Bsal/M36 family 3
        '#7895C1',  # Bsal/M36 family 2
        '#8074C8'   # Bsal/M36 family 1 (最深)
    ]
    
    if style == 'discrete':
        cmap = ListedColormap(blue_hex[start_map_index:end_map_index if end_map_index is not None else len(blue_hex)], name='blue_family_cmap')
    elif style == 'gradient':
        cmap = LinearSegmentedColormap.from_list('blue_family_gradient', blue_hex[start_map_index:end_map_index], N=256)
    else:
        raise ValueError("style 参数仅支持 'discrete' 或 'gradient'")
    
    return cmap

def get_red_color_map(style='discrete'):
    """
    获取红色系配色映射（对应原图中剩余的所有类别）
    :param style: 色图模式，可选 'discrete'（离散色图）或 'gradient'（渐变插值色图）
    :return: matplotlib.colors.Colormap 对象
    """
    # 按“浅→深”顺序排列（数值低→浅，数值高→深）
    red_hex = [
        '#F7FBC9',  # Bsal/M36 family 6 (最浅)
        '#F5EBAE',  # Singleton outliers
        '#F0C284',  # Eh/M36 family
        '#EF8B67',  # Bsal, Bd, Eh and Hp M36 family
        '#E3625D',  # Bsal and Bd M36 family
        '#B54764',  # Bsal/M36 family (dark red)
        '#992224'   # Bsal/M36 family (darkest red) (最深)
    ]
    
    if style == 'discrete':
        cmap = ListedColormap(red_hex, name='red_family_cmap')
    elif style == 'gradient':
        cmap = LinearSegmentedColormap.from_list('red_family_gradient', red_hex, N=256)
    else:
        raise ValueError("style 参数仅支持 'discrete' 或 'gradient'")
    
    return cmap

def get_gray_to_red_color_map(style='gradient'):
    """
    获取从灰色到红色系的渐变色图（用于表示强度变化）
    :param style: 色图模式，可选 'discrete'（离散色图）或 'gradient'（渐变插值色图）
    :return: matplotlib.colors.Colormap 对象
    """
    # 从灰色逐步过渡到红色系
    gray_to_red_hex = [
        '#C0C0C0',  # 浅灰色（起始）
        '#A0A0A0',  # 中灰色
        '#808080',  # 深灰色
        '#F7FBC9',  # 浅黄色（过渡）
        '#F5EBAE',  # 黄色
        '#F0C284',  # 橙黄色
        '#EF8B67',  # 橙色
        '#E3625D',  # 浅红色
        '#B54764',  # 中红色
        '#992224'   # 深红色（结束）
    ]
    
    if style == 'discrete':
        cmap = ListedColormap(gray_to_red_hex, name='gray_to_red_cmap')
    elif style == 'gradient':
        cmap = LinearSegmentedColormap.from_list('gray_to_red_gradient', gray_to_red_hex, N=256)
    else:
        raise ValueError("style 参数仅支持 'discrete' 或 'gradient'")
    
    return cmap

def get_viridis_color_map(start_gray=0.2):
    """
    获取自定义灰色梯度色图（用于3D PSD可视化）
    :param start_gray: 起始灰度值（0-1），值越大越浅
    :return: matplotlib.colors.Colormap 对象
    """
    import numpy as np
    start_gray = np.clip(start_gray, 0.0, 1.0)
    original_cmap = plt.cm.gist_yarg
    colors = original_cmap(np.linspace(start_gray, 1.0, 256))
    return ListedColormap(colors)


# ------------------------------
# 测试代码（可选）
# ------------------------------
if __name__ == "__main__":
    # 获取三种色系的色图
    cmap_blue_discrete = get_blue_color_map(style='discrete')
    cmap_blue_gradient = get_blue_color_map(style='gradient')
    cmap_red_discrete = get_red_color_map(style='discrete')
    cmap_red_gradient = get_red_color_map(style='gradient')
    cmap_gray_to_red_discrete = get_gray_to_red_color_map(style='discrete')
    cmap_gray_to_red_gradient = get_gray_to_red_color_map(style='gradient')
    
    # 可视化色图效果
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 蓝色系离散色图
    axes[0,0].imshow([range(5)], cmap=cmap_blue_discrete, aspect='auto')
    axes[0,0].set_title("Blue Family - Discrete (5 Categories)")
    axes[0,0].set_xticks(range(5))
    axes[0,0].set_xticklabels([f'Cat {i+1}' for i in range(5)], rotation=45, ha='right')
    axes[0,0].set_yticks([])
    
    # 蓝色系渐变色图
    axes[0,1].imshow([range(256)], cmap=cmap_blue_gradient, aspect='auto')
    axes[0,1].set_title("Blue Family - Gradient (256 Colors)")
    axes[0,1].set_yticks([])
    
    # 红色系离散色图
    axes[1,0].imshow([range(7)], cmap=cmap_red_discrete, aspect='auto')
    axes[1,0].set_title("Red Family - Discrete (7 Categories)")
    axes[1,0].set_xticks(range(7))
    axes[1,0].set_xticklabels([f'Cat {i+1}' for i in range(7)], rotation=45, ha='right')
    axes[1,0].set_yticks([])
    
    # 红色系渐变色图
    axes[1,1].imshow([range(256)], cmap=cmap_red_gradient, aspect='auto')
    axes[1,1].set_title("Red Family - Gradient (256 Colors)")
    axes[1,1].set_yticks([])
    
    # 灰→红色系离散色图
    axes[2,0].imshow([range(10)], cmap=cmap_gray_to_red_discrete, aspect='auto')
    axes[2,0].set_title("Gray-to-Red Family - Discrete (10 Categories)")
    axes[2,0].set_xticks(range(10))
    axes[2,0].set_xticklabels([f'Cat {i+1}' for i in range(10)], rotation=45, ha='right')
    axes[2,0].set_yticks([])
    
    # 灰→红色系渐变色图
    axes[2,1].imshow([range(256)], cmap=cmap_gray_to_red_gradient, aspect='auto')
    axes[2,1].set_title("Gray-to-Red Family - Gradient (256 Colors)")
    axes[2,1].set_yticks([])
    
    plt.tight_layout()
    plt.show()



