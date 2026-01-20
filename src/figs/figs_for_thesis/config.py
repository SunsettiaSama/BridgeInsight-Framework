"""
绘图基础配置文件
从fig2_2_time_series_rms.py中提取的配置，作为后续绘图的基础配置
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ==================== 字号配置 ====================
FONT_SIZE = 16

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
# RMS_TRHESHOLD = 0.16 # RMS阈值已改为动态计算（95%分位值）

# 颜色配置（根据阈值区分）
BELOW_THRESHOLD_COLOR = '#8074C8'  # 小于标准差阈值的颜色
ABOVE_THRESHOLD_COLOR = '#7895C1'  # 大于标准差阈值的颜色

