import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = 22
datasets_root = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\train\data\\' 

def datasets_description():

    def analyze_mat_files(folder_path):
        """
        分析指定文件夹下所有.mat文件中的标签，仅统计一般振动/涡激振动（剔除过渡状态），并绘制2D饼状图。
        
        :param folder_path: 包含.mat文件的文件夹路径
        """
        label_counts = {}

        def int2label(num):
            """将数字标签映射为语义标签（仅保留有效类别，剔除过渡态）"""
            if num == 0:
                label = '一般振动样本'
            elif num == 2:
                label = '涡激振动样本'
            # 直接跳过过渡态（num=1），不返回任何标签
            return label if num in [0,2] else None

        # 遍历文件夹下的所有.mat文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(folder_path, filename)
                mat_contents = sio.loadmat(file_path)
                
                # 过滤mat文件的系统关键字，仅统计有效标签（剔除过渡态）
                for key in mat_contents.keys():
                    if key not in ['__globals__', '__header__', '__version__']:
                        try:
                            key_int = int(key)  # 确保key能转为数字
                            label = int2label(key_int)
                            if label is not None:  # 仅统计非过渡态标签
                                label_counts[label] = label_counts.get(label, 0) + 1
                        except ValueError:
                            # 跳过非数字的无效key
                            continue
        
        # 校验统计结果（防止无有效样本）
        if not label_counts:
            raise ValueError("过滤过渡态后无有效样本可统计！")
        
        # 准备绘图数据
        labels = list(label_counts.keys())
        sizes = list(label_counts.values())
        total_samples = np.sum(sizes)
        print(f"样本统计结果（剔除过渡态）：")
        for label, count in label_counts.items():
            print(f"{label}：{count}个（占比{count/total_samples*100:.1f}%）")

        # -------------------------- 适配两类样本的配色方案 --------------------------
        # 自定义配色（仅保留两类，匹配一般振动/涡激振动）
        custom_palette = [
            '#BFDFD2',  # 一般振动样本
            '#68BED9',  # 涡激振动样本
            '#257D8B',  # 原过渡态配色（弃用）
            '#EFCE87',  # 备用颜色
            '#EAA558',
            '#ED8D5A'
        ]
        # 按标签顺序取对应颜色（保证一般振动→#BFDFD2，涡激振动→#68BED9）
        colors = [custom_palette[i] for i in [2, -1]]

        # -------------------------- 2D饼图绘制（适配两类样本） --------------------------
        # 设置画布大小+高分辨率，适配论文配图
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # 饼块轻微偏移（两类样本各偏移0.05，增强层次感）
        explode = (0.05, 0.05) if len(labels)==2 else [0.05]*len(labels)
        # 绘制2D饼图（应用自定义配色）
        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct='%1.1f%%',  # 百分比显示格式
            startangle=140,     # 起始角度，避免标签重叠
            colors=colors,      # 适配两类的配色
            explode=explode,    # 饼块偏移
            shadow=True,        # 阴影增强立体感
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}  # 饼块白色描边
        )

        # -------------------------- 样式精细化优化 --------------------------
        # 百分比文本：白色加粗，更醒目
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 保证饼图为正圆形
        ax.axis('equal')

        # 图例：位置优化，适配两类样本
        ax.legend(
            wedges, 
            labels, 
            loc='upper right', 
            frameon=True,  # 显示图例边框
            fancybox=True  # 圆角图例框
        )

        # 自动调整布局，避免标题/标签被截断
        plt.tight_layout()
        # 可选：保存高清图片（用于论文，分辨率300dpi）
        # plt.savefig('sample_distribution_pie_no_transition.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 执行分析与绘图
    analyze_mat_files(datasets_root)

    return

# 调用函数
datasets_description()