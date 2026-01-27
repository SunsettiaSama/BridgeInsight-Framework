from .io_unpacker import UNPACK, DataManager, parse_path_str, parse_path_metadata

# 延迟导入以避免某些环境下的 NumPy/SciPy 版本冲突
# 如果环境正常，可以直接使用 from .algorithms import ...
def get_algorithms():
    from . import algorithms
    return algorithms

def get_persistence_utils():
    from . import persistence_utils
    return persistence_utils

def get_database_manager():
    from . import database_manager
    return database_manager

def get_pipeline_orchestrator():
    from . import pipeline_orchestrator
    return pipeline_orchestrator
