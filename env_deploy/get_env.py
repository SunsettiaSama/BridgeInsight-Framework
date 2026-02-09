import subprocess
import sys
import os
from typing import Optional, List

# ===================== 排除字段常量（可按需扩展）=====================
# 1. 需要排除的包名前缀（如 conda-、anaconda- 开头的包）
EXCLUDED_PREFIXES = (
    "conda-",          # 所有conda相关核心包
    "anaconda-",       # 所有anaconda专属包
    "ruamel-yaml-conda"# conda定制的yaml包
)

# 2. 需要排除的完整包名（精确匹配）
EXCLUDED_NAMES = (
    "spyder",          # Anaconda默认IDE
    "spyder-kernels",  # spyder配套内核
    "libmambapy",      # Conda依赖的libmamba绑定
    "menuinst",        # Anaconda菜单安装工具
    "navigator-updater",# Anaconda Navigator更新工具
    "uv"               # 可选：排除uv本身（如果不需要）
    "black",
    "urllib3",
    "clyent", 
    
)

def check_uv_installed() -> bool:
    """
    检查当前环境是否安装了 uv 工具
    返回：True 表示已安装，False 表示未安装
    """
    try:
        # 执行 uv --version 命令，静默输出（不显示到控制台）
        result = subprocess.run(
            [sys.executable, "-m", "uv", "--version"] if os.name == 'nt' else ["uv", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ 检测到 uv 已安装，版本：{result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        # uv 存在但执行失败
        print("❌ uv 已安装但执行失败，请检查 uv 版本是否正常")
        return False
    except FileNotFoundError:
        # 未找到 uv 命令
        print("❌ 未检测到 uv 工具，请先安装 uv：pip install uv")
        return False

def filter_excluded_packages(raw_lines: List[str]) -> List[str]:
    """
    过滤掉Anaconda/Conda相关的内置包
    参数：raw_lines - uv pip list 输出的原始行列表
    返回：过滤后的行列表
    """
    filtered_lines = []
    excluded_count = 0  # 统计被排除的包数量
    
    for line in raw_lines:
        # 跳过空行
        line = line.strip()
        if not line:
            continue
        
        # 拆分包名和版本（freeze格式：包名==版本号）
        package_name = line.split("==")[0].lower()  # 转小写避免大小写问题
        
        # 检查是否匹配排除前缀或排除名称
        is_excluded = False
        # 检查前缀
        if any(package_name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            is_excluded = True
        # 检查完整名称
        elif package_name in (name.lower() for name in EXCLUDED_NAMES):
            is_excluded = True
        
        if is_excluded:
            excluded_count += 1
            print(f"🔍 排除Anaconda内置包：{line}")
        else:
            filtered_lines.append(line)
    
    print(f"📋 共排除 {excluded_count} 个Anaconda内置包")
    return filtered_lines

def generate_requirements_file(output_path: str = "requirements.txt") -> Optional[bool]:
    """
    执行 uv pip list --format=freeze 命令，过滤后写入 requirements.txt
    参数：output_path - 输出文件路径，默认是当前目录的 requirements.txt
    返回：True 成功，False 失败，None 前置检查不通过
    """
    # 前置检查：确保 uv 已安装
    if not check_uv_installed():
        return None
    
    try:
        # 核心命令：uv pip list --format=freeze
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "list", "--format=freeze"] if os.name == 'nt' else ["uv", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"  # 指定编码，避免中文/特殊字符乱码
        )
        
        # 拆分原始输出为行列表
        raw_lines = result.stdout.splitlines()
        print(f"📥 从uv获取到 {len(raw_lines)} 个原始依赖包")
        
        # 过滤Anaconda内置包
        filtered_lines = filter_excluded_packages(raw_lines)
        
        # 将过滤后的内容写入文件（覆盖原有内容）
        with open(output_path, "w", encoding="utf-8") as f:
            # 每行一个包，保持freeze格式
            f.write("\n".join(filtered_lines))
        
        # 验证文件是否生成并输出统计信息
        if os.path.exists(output_path):
            print(f"✅ 过滤后的依赖文件生成成功！路径：{os.path.abspath(output_path)}")
            print(f"📊 最终导出 {len(filtered_lines)} 个第三方依赖包")
            return True
        else:
            print("❌ 依赖文件生成失败：文件未创建")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行 uv 命令失败，错误码：{e.returncode}")
        print(f"错误信息：{e.stderr.strip()}")
        return False
    except PermissionError:
        print(f"❌ 权限不足，无法写入文件：{output_path}")
        return False
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")
        return False

if __name__ == "__main__":
    # 主程序入口
    print("===== 开始导出并过滤当前环境依赖 =====")
    # 执行核心功能，生成 requirements.txt
    success = generate_requirements_file()
    # 根据执行结果退出程序
    sys.exit(0 if success else 1)