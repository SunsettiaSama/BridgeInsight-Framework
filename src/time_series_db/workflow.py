

from .table import FileTypeMappingConfig, TimeSeriesFileIndex
from typing import Optional, List
from pathlib import Path

def get_index_table(index_path: str = r'F:\Research\Vibration Characteristics In Cable Vibration\data\index.parquet',
                   config_path: Optional[str] = r'F:\Research\Vibration Characteristics In Cable Vibration\data\config.yaml',
                   file_paths: Optional[List[str]] = None,
                   save_dir: Optional[str] = None) -> TimeSeriesFileIndex:
    """
    获取或创建时序文件索引数据库
    
    Args:
        index_path: 索引Parquet文件路径。如果文件不存在，将创建新索引。
        config_path: 配置文件路径。如果为None，将使用默认配置。
        file_paths: 要添加到索引的文件路径列表（可选）
        save_dir: 保存更新后索引的目录（可选）
    
    Returns:
        加载并可能更新后的TimeSeriesFileIndex实例
    """
    print(f"正在加载索引数据库...")
    
    # 检查索引文件是否存在
    index_exists = Path(index_path).exists() if index_path else False
    
    if index_exists:
        try:
            # 加载现有索引
            db = TimeSeriesFileIndex.load_from_parquet(index_path, config_path)
            print(f"✅ 成功加载现有索引: {index_path}")
        except Exception as e:
            print(f"❌ 加载索引失败: {str(e)}，将创建新索引")
            db = TimeSeriesFileIndex()
    else:
        print(f"ℹ️ 索引文件不存在，创建新索引")
        db = TimeSeriesFileIndex()
        
        # 如果提供了配置路径，加载配置
        if config_path and Path(config_path).exists():
            try:
                db.mapping_config = FileTypeMappingConfig(config_path)
                print(f"✅ 加载配置文件: {config_path}")
            except Exception as e:
                print(f"⚠️ 加载配置文件失败，使用默认配置: {str(e)}")
    
    # 如果提供了文件路径，添加文件到索引
    if file_paths:
        print(f"正在添加 {len(file_paths)} 个文件到索引...")
        db.add_files(file_paths)
        print(f"✅ 文件添加完成，当前索引包含 {len(db.df)} 条记录")
    
    # 如果提供了保存目录，保存索引
    if save_dir:
        print(f"正在保存索引到目录: {save_dir}")
        db.save_to_parquet(save_dir)
        print(f"✅ 索引已保存至 {save_dir}/index.parquet 和 {save_dir}/config.yaml")
    
    print("\n" + "="*60)
    print("索引数据库摘要:")
    print("="*60)
    print(db)
    
    return db
