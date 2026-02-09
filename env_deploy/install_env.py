import subprocess
import sys
import os
import logging
from typing import List, Dict
from datetime import datetime
import chardet  # 新增：编码检测库

# ===================== 配置常量（可按需调整）=====================
# 输入的requirements文件路径
REQUIREMENTS_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\env_deploy\requirements.txt"
# 安装失败的明细日志文件路径
FAILED_LOG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\env_deploy\failed_install_packages.log"
# 脚本整体运行日志文件路径
RUN_LOG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\env_deploy\install_requirements_run.log"
# uv安装命令超时时间（秒），避免卡壳
INSTALL_TIMEOUT = 300

# ===================== 日志配置 =====================
def setup_logging() -> None:
    """
    配置日志系统：同时输出到控制台和运行日志文件
    """
    # 日志格式：时间 | 级别 | 消息
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 清除原有日志处理器（避免重复输出）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,  # 输出INFO及以上级别日志
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台处理器（彩色输出优化，可选）
            logging.StreamHandler(sys.stdout),
            # 文件处理器（写入运行日志）
            logging.FileHandler(RUN_LOG_PATH, encoding="utf-8", mode="w")
        ]
    )

def check_uv_installed() -> bool:
    """
    检查当前环境是否安装了 uv 工具
    返回：True 表示已安装，False 表示未安装
    """
    try:
        cmd = [sys.executable, "-m", "uv", "--version"] if os.name == 'nt' else ["uv", "--version"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        logging.info(f"检测到 uv 已安装，版本：{result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        logging.error("uv 已安装但执行失败，请检查 uv 版本是否正常")
        return False
    except FileNotFoundError:
        logging.error("未检测到 uv 工具，请先安装 uv：pip install uv")
        return False
    except Exception as e:
        logging.error(f"检查 uv 失败：{str(e)}")
        return False

def detect_file_encoding(file_path: str) -> str:
    """
    检测文件的编码格式
    参数：file_path - 文件路径
    返回：检测到的编码字符串（如 'utf-8', 'utf-16-le', 'gbk' 等）
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        # 检测编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        logging.info(f"检测到文件编码：{encoding}（置信度：{confidence:.2f}）")
        # 兼容处理：将 utf-16-le 转为 utf-16（避免解码问题）
        if encoding == 'utf-16-le':
            encoding = 'utf-16'
        return encoding or 'utf-8'  # 兜底用UTF-8
    except Exception as e:
        logging.warning(f"检测文件编码失败，默认使用UTF-8：{str(e)}")
        return 'utf-8'

def parse_requirements(file_path: str) -> List[str]:
    """
    解析requirements.txt文件（自动兼容多编码），提取有效包名
    参数：file_path - requirements文件路径
    返回：有效包名列表（格式：包名==版本号）
    """
    valid_packages = []
    if not os.path.exists(file_path):
        logging.error(f"未找到requirements文件：{os.path.abspath(file_path)}")
        return valid_packages
    
    try:
        # 步骤1：检测文件编码
        file_encoding = detect_file_encoding(file_path)
        
        # 步骤2：用检测到的编码读取文件
        with open(file_path, "r", encoding=file_encoding) as f:
            lines = f.readlines()
        
        # 步骤3：解析有效包名（逻辑不变）
        for line_num, line in enumerate(lines, 1):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue
            if any(prefix in clean_line for prefix in ("git+", "file://", "./", "/")):
                logging.warning(f"第{line_num}行：非标准包格式，跳过 -> {clean_line}")
                continue
            valid_packages.append(clean_line)
        
        logging.info(f"成功解析requirements文件，共提取 {len(valid_packages)} 个有效包")
        return valid_packages
    
    except Exception as e:
        # 兜底：尝试多种常见编码重试
        logging.warning(f"用 {file_encoding} 编码读取失败，尝试常见编码重试：{str(e)}")
        common_encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
        for encoding in common_encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    lines = f.readlines()
                logging.info(f"使用 {encoding} 编码读取文件成功")
                # 重新解析
                valid_packages = []
                for line_num, line in enumerate(lines, 1):
                    clean_line = line.strip()
                    if not clean_line or clean_line.startswith("#"):
                        continue
                    if any(prefix in clean_line for prefix in ("git+", "file://", "./", "/")):
                        logging.warning(f"第{line_num}行：非标准包格式，跳过 -> {clean_line}")
                        continue
                    valid_packages.append(clean_line)
                logging.info(f"成功解析requirements文件，共提取 {len(valid_packages)} 个有效包")
                return valid_packages
            except Exception as e2:
                logging.warning(f"尝试 {encoding} 编码失败：{str(e2)}")
        
        # 所有编码都失败
        logging.error(f"解析requirements文件失败：所有常见编码均无法解码文件")
        return valid_packages

def install_packages(packages: List[str]) -> Dict[str, str]:
    """
    逐个安装包，记录安装失败的包和错误信息
    参数：packages - 待安装的包列表
    返回：失败的包字典 {包名: 错误信息}
    """
    failed_packages = {}
    total = len(packages)
    
    logging.info(f"开始安装 {total} 个依赖包")
    for idx, package in enumerate(packages, 1):
        logging.info(f"[{idx}/{total}] 正在安装：{package}")
        try:
            # 构建uv安装命令（逐个安装，避免一个失败影响全部）
            cmd = [sys.executable, "-m", "uv", "pip", "install", package] if os.name == 'nt' else ["uv", "pip", "install", package]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=INSTALL_TIMEOUT,
                encoding="utf-8"
            )
            logging.info(f"安装成功：{package}")
        except subprocess.CalledProcessError as e:
            # 安装返回非0错误码（安装失败）
            error_msg = e.stderr.strip() if e.stderr else e.stdout.strip()
            # 截取错误信息前500字符，避免日志过长
            short_error = error_msg[:500] + "..." if len(error_msg) > 500 else error_msg
            failed_packages[package] = short_error
            logging.warning(f"安装失败（已跳过）：{package}")
            logging.warning(f"   错误原因：{short_error}")
        except subprocess.TimeoutExpired:
            error_msg = f"安装超时（超过{INSTALL_TIMEOUT}秒）"
            failed_packages[package] = error_msg
            logging.warning(f"安装失败（已跳过）：{package}")
            logging.warning(f"   错误原因：{error_msg}")
        except Exception as e:
            error_msg = str(e)
            failed_packages[package] = error_msg
            logging.warning(f"安装失败（已跳过）：{package}")
            logging.warning(f"   错误原因：{error_msg}")
    
    return failed_packages

def write_failed_log(failed_pkgs: Dict[str, str], log_path: str) -> bool:
    """
    将安装失败的包写入明细日志文件
    参数：failed_pkgs - 失败包字典，log_path - 日志文件路径
    返回：True 成功，False 失败
    """
    if not failed_pkgs:
        logging.info("所有包均安装成功，无需生成失败明细日志")
        return True
    
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("===== 依赖包安装失败明细日志 =====\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"失败总数：{len(failed_pkgs)}\n\n")
            for pkg, error in failed_pkgs.items():
                f.write(f"包名：{pkg}\n")
                f.write(f"错误原因：{error}\n")
                f.write("-" * 80 + "\n")
        
        logging.info(f"失败明细日志已生成：{os.path.abspath(log_path)}")
        logging.error(f"共 {len(failed_pkgs)} 个包安装失败，详见明细日志文件")
        return True
    except Exception as e:
        logging.error(f"写入失败明细日志失败：{str(e)}")
        return False

def main():
    """主程序入口"""
    # 初始化日志配置
    setup_logging()
    logging.info("===== 依赖包批量安装验证脚本启动 =====")
    
    # 步骤1：检查uv是否安装
    if not check_uv_installed():
        logging.critical("uv检查失败，程序退出")
        sys.exit(1)
    
    # 步骤2：解析requirements文件
    packages = parse_requirements(REQUIREMENTS_PATH)
    if not packages:
        logging.critical("无有效包可安装，程序退出")
        sys.exit(1)
    
    # 步骤3：逐个安装包，记录失败
    failed_pkgs = install_packages(packages)
    
    # 步骤4：写入失败明细日志
    write_failed_log(failed_pkgs, FAILED_LOG_PATH)
    
    # 步骤5：汇总结果
    success_count = len(packages) - len(failed_pkgs)
    logging.info("===== 安装完成汇总 =====")
    logging.info(f"总待安装包数：{len(packages)}")
    logging.info(f"安装成功数：{success_count}")
    logging.info(f"安装失败数：{len(failed_pkgs)}")
    
    sys.exit(0 if not failed_pkgs else 1)

if __name__ == "__main__":
    main()