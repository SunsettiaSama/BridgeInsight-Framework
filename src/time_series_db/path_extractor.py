import os
import logging
from collections import deque
from typing import List, Generator, Optional, Dict, Any, Union, Set, Tuple
from pathlib import Path
import re
import datetime

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 类型别名
DirectoryStructure = Dict[str, Union[Dict[str, Any], List[str]]]
PathFilter = Optional[callable]

def _normalize_path(path: str) -> str:
    """标准化路径，转为绝对路径并统一使用正斜杠"""
    return os.path.abspath(path).replace('\\', '/')

def _should_exclude(item_name: str, exclude_hidden: bool) -> bool:
    """判断是否应该排除该项目"""
    return exclude_hidden and item_name.startswith('.')

def get_all_files_list(root_dir: str, 
                        exclude_hidden: bool = True,
                        follow_symlinks: bool = False,
                        max_depth: Optional[int] = None,
                        path_filter: "PathFilter" = None) -> List[str]:
    """
    使用os.walk获取目录下所有文件的绝对路径（推荐使用）
    
    Args:
        root_dir: 根目录路径
        exclude_hidden: 是否排除隐藏文件和文件夹（以.开头）
        follow_symlinks: 是否跟随符号链接
        max_depth: 最大遍历深度，None表示无限制
        path_filter: 自定义过滤函数，接收绝对路径，返回True表示保留
        
    Returns:
        文件绝对路径列表，统一使用正斜杠
        
    Raises:
        FileNotFoundError: 当目录不存在时
        PermissionError: 当没有访问权限时
        ValueError: 当传入无效路径时
    """
    if not root_dir:
        raise ValueError("root_dir cannot be empty")
        
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not os.path.isdir(root_dir):
        raise ValueError(f"Path is not a directory: {root_dir}")
    
    file_paths = []
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_symlinks):
            # 计算当前深度
            current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            
            # 检查深度限制
            if max_depth is not None and current_depth > max_depth:
                dirnames[:] = []  # 清空防止继续遍历
                continue
                
            # 过滤隐藏目录
            if exclude_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            
            for filename in filenames:
                if _should_exclude(filename, exclude_hidden):
                    continue
                
                full_path = os.path.join(dirpath, filename)
                abs_path = _normalize_path(full_path)
                
                # 应用自定义过滤
                if path_filter and not path_filter(abs_path):
                    continue
                    
                file_paths.append(abs_path)
                
    except PermissionError as e:
        logger.warning(f"Permission denied accessing {e.filename}, skipping.")
    except Exception as e:
        logger.error(f"Error while traversing directory: {e}")
        raise
    
    return file_paths


def get_all_files_generator(root_dir: str,
                          exclude_hidden: bool = True,
                          follow_symlinks: bool = False,
                          max_depth: Optional[int] = None) -> Generator[str, None, None]:
    """
    使用生成器方式遍历文件，节省内存
    
    Args:
        root_dir: 根目录路径
        exclude_hidden: 是否排除隐藏文件和文件夹
        follow_symlinks: 是否跟随符号链接
        max_depth: 最大遍历深度，None表示无限制
        
    Yields:
        文件绝对路径，统一使用正斜杠
        
    Raises:
        FileNotFoundError: 当目录不存在时
        ValueError: 当传入无效路径时
    """
    if not root_dir:
        raise ValueError("root_dir cannot be empty")
        
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not os.path.isdir(root_dir):
        raise ValueError(f"Path is not a directory: {root_dir}")
    
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_symlinks):
            current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            
            if max_depth is not None and current_depth > max_depth:
                dirnames[:] = []
                continue
                
            if exclude_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            
            for filename in filenames:
                if _should_exclude(filename, exclude_hidden):
                    continue
                    
                full_path = os.path.join(dirpath, filename)
                yield _normalize_path(full_path)
    except PermissionError as e:
        logger.warning(f"Permission denied accessing {e.filename}, skipping.")
    except Exception as e:
        logger.error(f"Error while traversing directory: {e}")
        raise


def get_files_by_extension(root_dir: str, 
                          extensions: Optional[List[str]] = None,
                          exclude_hidden: bool = True,
                          follow_symlinks: bool = False,
                          max_depth: Optional[int] = None) -> List[str]:
    """
    根据文件扩展名过滤文件
    
    Args:
        root_dir: 根目录路径
        extensions: 需要的文件扩展名列表，如 ['.py', '.txt']，None表示所有文件
        exclude_hidden: 是否排除隐藏文件
        follow_symlinks: 是否跟随符号链接
        max_depth: 最大遍历深度
        
    Returns:
        符合条件的文件路径列表
    """
    # 处理扩展名格式
    valid_extensions: Optional[Set[str]] = None
    if extensions is not None:
        valid_extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}
    
    file_paths = []
    root_dir = os.path.abspath(root_dir)
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_symlinks):
        current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        
        if max_depth is not None and current_depth > max_depth:
            dirnames[:] = []
            continue
            
        if exclude_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        for filename in filenames:
            if _should_exclude(filename, exclude_hidden):
                continue
            
            # 提前过滤，避免不必要的路径计算
            if valid_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in valid_extensions:
                    continue
            
            full_path = os.path.join(dirpath, filename)
            file_paths.append(_normalize_path(full_path))
            
    return file_paths


def get_directory_structure(root_dir: str, 
                          exclude_hidden: bool = True,
                          max_depth: Optional[int] = None) -> DirectoryStructure:
    """
    获取目录结构，返回嵌套字典形式
    
    Args:
        root_dir: 根目录路径
        exclude_hidden: 是否排除隐藏文件
        max_depth: 最大遍历深度
        
    Returns:
        目录结构字典，格式:
        {
            "name": "root_dir_name",
            "path": "absolute_path",
            "directories": [  # 子目录列表
                {
                    "name": "subdir_name",
                    "path": "absolute_path",
                    "directories": [...],
                    "files": ["file1.txt", ...]
                }
            ],
            "files": ["rootfile1.txt", ...]
        }
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not os.path.isdir(root_dir):
        raise ValueError(f"Path is not a directory: {root_dir}")
    
    root_name = os.path.basename(root_dir)
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    
    def _build_structure(current_path: str, current_depth: int = 0) -> Dict[str, Any]:
        """递归构建目录结构"""
        structure = {
            "name": os.path.basename(current_path),
            "path": _normalize_path(current_path),
            "directories": [],
            "files": []
        }
        
        try:
            entries = os.listdir(current_path)
        except PermissionError:
            logger.warning(f"Permission denied: {current_path}")
            return structure
        
        # 分离目录和文件
        subdirs = []
        files = []
        
        for entry in entries:
            if _should_exclude(entry, exclude_hidden):
                continue
                
            full_path = os.path.join(current_path, entry)
            try:
                if os.path.isdir(full_path) and not os.path.islink(full_path):
                    subdirs.append(entry)
                elif os.path.isfile(full_path):
                    files.append(entry)
            except Exception as e:
                logger.warning(f"Error processing {full_path}: {e}")
        
        # 处理文件
        structure["files"] = sorted(files)
        
        # 递归处理子目录
        if max_depth is None or current_depth < max_depth:
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(current_path, subdir)
                structure["directories"].append(_build_structure(subdir_path, current_depth + 1))
        
        return structure
    
    return _build_structure(root_dir)


def count_files_by_type(root_dir: str, 
                       exclude_hidden: bool = True,
                       follow_symlinks: bool = False,
                       max_depth: Optional[int] = None) -> Dict[str, int]:
    """
    统计各类文件的数量
    
    Args:
        root_dir: 根目录路径
        exclude_hidden: 是否排除隐藏文件
        follow_symlinks: 是否跟随符号链接
        max_depth: 最大遍历深度
        
    Returns:
        文件类型到数量的映射字典
    """
    file_types = {}
    
    for file_path in get_all_files_generator(root_dir, exclude_hidden, follow_symlinks, max_depth):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower() if ext else '无扩展名'
        file_types[ext] = file_types.get(ext, 0) + 1
    
    return file_types


# 还需要分类和查找的，能够自动归类的文件
# 按照我们之前的构想，这个归类文件应当能够依照时间进行排序
# 而这一归类还得能够解析上下游，风速等不同的path字段，与时间对应

def extract_metadata_from_path(file_path: str) -> dict:
    """
    从文件路径中提取元数据
    
    Args:
        file_path: 文件路径字符串
    
    Returns:
        包含元数据的字典，包含时间信息和传感器信息
    """
    # 初始化结果字典
    metadata = {
        "time": {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
            "second": None,
            "datetime": None
        },
        "sensor": {
            "name": "",
            "type": ""
        }
    }
    
    # 转换为Path对象
    path = Path(file_path)
    
    # 1. 提取传感器类型 (文件扩展名)
    if path.suffix:
        metadata["sensor"]["type"] = path.suffix[1:]  # 去掉点号
    
    # 2. 从文件名中提取传感器名称和时间
    filename = path.stem  # 不含扩展名的文件名
    
    # 查找时间部分 (格式为 _000000)
    time_match = re.search(r'_(\d{6})$', filename)
    if time_match:
        time_str = time_match.group(1)
        metadata["time"]["hour"] = int(time_str[0:2])
        metadata["time"]["minute"] = int(time_str[2:4])
        metadata["time"]["second"] = int(time_str[4:6])
        
        # 传感器名称是文件名去掉时间部分
        metadata["sensor"]["name"] = filename.replace(f'_{time_str}', '')
    
    # 3. 从路径中提取日期 (test_data后的两个目录)
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() == "test_data" and i + 2 < len(parts):
            try:
                metadata["time"]["month"] = int(parts[i + 1])
                metadata["time"]["day"] = int(parts[i + 2])
            except ValueError:
                pass
            break
    
    # 4. 尝试构造完整的时间
    if metadata["time"]["month"] and metadata["time"]["day"]:
        # 使用当前年份作为默认值
        current_year = datetime.datetime.now().year
        metadata["time"]["year"] = current_year
        
        try:
            dt = datetime.datetime(
                year=metadata["time"]["year"],
                month=metadata["time"]["month"],
                day=metadata["time"]["day"],
                hour=metadata["time"]["hour"] or 0,
                minute=metadata["time"]["minute"] or 0,
                second=metadata["time"]["second"] or 0
            )
            metadata["time"]["datetime"] = dt
        except ValueError:
            pass
    
    return metadata


