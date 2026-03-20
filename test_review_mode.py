#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试annotation复盘模式
"""

from src.visualize_tools.annotation_tools.annotation import AnnotationGUI, AnnotationDataProvider

if __name__ == "__main__":
    print("=" * 60)
    print("测试 annotation 复盘模式")
    print("=" * 60)
    
    # 测试1: 检查复盘模式常量是否存在
    print("\n[测试1] 检查复盘模式常量")
    print(f"MODE_NORMAL: {AnnotationDataProvider.MODE_NORMAL}")
    print(f"MODE_EXTREME: {AnnotationDataProvider.MODE_EXTREME}")
    print(f"MODE_SUPER_EXTREME: {AnnotationDataProvider.MODE_SUPER_EXTREME}")
    print(f"MODE_REVIEW: {AnnotationDataProvider.MODE_REVIEW}")
    
    # 启动GUI
    print("\n[启动] 启动 annotation GUI")
    print("在模式选择对话框中选择 '复盘模式' 来测试新功能")
    print("\n新功能说明:")
    print("  1. 在开始菜单中添加了'复盘模式'选项")
    print("  2. 复盘模式下只加载已标注过的样本")
    print("  3. 预填已有的标注，光标在末尾")
    print("  4. 添加了'总览'按钮显示元数据")
    print("  5. 复盘模式下不会自动跳过窗口")
    print("-" * 60)
    
    app = AnnotationGUI()
    app.run()
