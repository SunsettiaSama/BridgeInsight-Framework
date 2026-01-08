"""
配置管理模块 - 修复版本
提供类型安全的配置管理，支持从YAML/JSON加载和验证
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union, Tuple, Set
from pathlib import Path
import yaml
import json
import logging
import re
from enum import Enum

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    cache_dir: str = "./cache"
    auto_save: bool = True
    validate_cache: bool = True
    compression: bool = False
    
    def validate(self) -> None:
        """验证缓存配置"""
        if not isinstance(self.enabled, bool):
            raise ValueError("cache.enabled must be a boolean")
        if not isinstance(self.cache_dir, str):
            raise ValueError("cache.cache_dir must be a string")
        if not isinstance(self.auto_save, bool):
            raise ValueError("cache.auto_save must be a boolean")
        if not isinstance(self.validate_cache, bool):
            raise ValueError("cache.validate_cache must be a boolean")
        if not isinstance(self.compression, bool):
            raise ValueError("cache.compression must be a boolean")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """从字典创建CacheConfig实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class ProcessingConfig:
    """处理配置"""
    use_parallel: bool = True
    batch_size: int = 1000
    max_workers: Optional[int] = None
    exclude_hidden: bool = True
    follow_symlinks: bool = False
    max_depth: Optional[int] = None
    use_generator: bool = True
    memory_threshold_mb: int = 2000
    progress_bar: bool = True
    pre_scan_total_files: bool = True
    chunk_size: int = 10000
    
    def validate(self) -> None:
        """验证处理配置"""
        if not isinstance(self.use_parallel, bool):
            raise ValueError("processing.use_parallel must be a boolean")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("processing.batch_size must be a positive integer")
        if self.max_workers is not None and (not isinstance(self.max_workers, int) or self.max_workers <= 0):
            raise ValueError("processing.max_workers must be a positive integer or null")
        if not isinstance(self.exclude_hidden, bool):
            raise ValueError("processing.exclude_hidden must be a boolean")
        if not isinstance(self.follow_symlinks, bool):
            raise ValueError("processing.follow_symlinks must be a boolean")
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth <= 0):
            raise ValueError("processing.max_depth must be a positive integer or null")
        if not isinstance(self.use_generator, bool):
            raise ValueError("processing.use_generator must be a boolean")
        if not isinstance(self.memory_threshold_mb, int) or self.memory_threshold_mb <= 0:
            raise ValueError("processing.memory_threshold_mb must be a positive integer")
        if not isinstance(self.progress_bar, bool):
            raise ValueError("processing.progress_bar must be a boolean")
        if not isinstance(self.pre_scan_total_files, bool):
            raise ValueError("processing.pre_scan_total_files must be a boolean")
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError("processing.chunk_size must be a positive integer")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """从字典创建ProcessingConfig实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class SensorConfig:
    """传感器配置"""
    enabled: bool = True
    pattern: str = r".*"
    description: str = ""
    
    def validate(self) -> None:
        """验证传感器配置"""
        if not isinstance(self.enabled, bool):
            raise ValueError("sensor.enabled must be a boolean")
        if not isinstance(self.pattern, str):
            raise ValueError("sensor.pattern must be a string")
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        if not isinstance(self.description, str):
            raise ValueError("sensor.description must be a string")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorConfig":
        """从字典创建SensorConfig实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class OutputConfig:
    """输出配置"""
    auto_generate_report: bool = True
    report_dir: str = "./reports"
    save_failed_files: bool = True
    detailed_statistics: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "txt"])
    
    def validate(self) -> None:
        """验证输出配置"""
        if not isinstance(self.auto_generate_report, bool):
            raise ValueError("output.auto_generate_report must be a boolean")
        if not isinstance(self.report_dir, str):
            raise ValueError("output.report_dir must be a string")
        if not isinstance(self.save_failed_files, bool):
            raise ValueError("output.save_failed_files must be a boolean")
        if not isinstance(self.detailed_statistics, bool):
            raise ValueError("output.detailed_statistics must be a boolean")
        if not isinstance(self.export_formats, list):
            raise ValueError("output.export_formats must be a list")
        valid_formats = {"json", "txt", "csv", "yaml"}
        if not set(self.export_formats).issubset(valid_formats):
            raise ValueError(f"output.export_formats must be a subset of {valid_formats}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """从字典创建OutputConfig实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class LoggingConfig:
    """日志配置"""
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = None
    
    def validate(self) -> None:
        """验证日志配置"""
        if not isinstance(self.level, LogLevel):
            raise ValueError(f"logging.level must be one of {list(LogLevel)}")
        if self.file is not None and not isinstance(self.file, str):
            raise ValueError("logging.file must be a string or null")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """从字典创建LoggingConfig实例"""
        # 处理日志级别转换
        level_data = data.get('level', 'INFO')
        if isinstance(level_data, str):
            try:
                level = LogLevel[level_data.upper()]
            except KeyError:
                level = LogLevel.INFO
        else:
            level = LogLevel.INFO
        
        return cls(
            level=level,
            file=data.get('file')
        )

@dataclass
class WorkflowConfig:
    """
    时序数据工作流主配置
    提供类型安全的配置管理
    """
    data_roots: List[str] = field(default_factory=list)
    cache: CacheConfig = field(default_factory=CacheConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """初始化后处理，设置默认传感器配置"""
        if not self.sensors:
            self.sensors = {
                "UAN": SensorConfig(enabled=True, pattern=r".*\.UAN$", description="Underwater Acoustic Noise"),
                "ACC": SensorConfig(enabled=True, pattern=r".*\.ACC$", description="Accelerometer"),
                "STRAIN": SensorConfig(enabled=True, pattern=r".*\.STRAIN$", description="Strain Gauge")
            }
    
    def validate(self) -> None:
        """验证整个配置"""
        # 验证data_roots
        if not isinstance(self.data_roots, list):
            raise ValueError("data_roots must be a list")
        for root in self.data_roots:
            if not isinstance(root, str):
                raise ValueError("Each data root must be a string")
        
        # 验证子配置
        if not isinstance(self.cache, CacheConfig):
            raise ValueError("cache must be a CacheConfig instance")
        if not isinstance(self.processing, ProcessingConfig):
            raise ValueError("processing must be a ProcessingConfig instance")
        if not isinstance(self.output, OutputConfig):
            raise ValueError("output must be an OutputConfig instance")
        if not isinstance(self.logging, LoggingConfig):
            raise ValueError("logging must be a LoggingConfig instance")
        
        self.cache.validate()
        self.processing.validate()
        self.output.validate()
        self.logging.validate()
        
        # 验证传感器配置
        if not isinstance(self.sensors, dict):
            raise ValueError("sensors must be a dictionary")
        for sensor_name, sensor_config in self.sensors.items():
            if not isinstance(sensor_name, str):
                raise ValueError("Sensor names must be strings")
            if not isinstance(sensor_config, SensorConfig):
                raise ValueError(f"Sensor {sensor_name} configuration must be a SensorConfig instance")
            sensor_config.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        result = asdict(self)
        # 转换LogLevel枚举为字符串
        result['logging']['level'] = self.logging.level.value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WorkflowConfig":
        """从字典创建配置实例 - 修复版本"""
        # 创建配置字典的副本，避免修改原始数据
        config_data = config_dict.copy()
        
        # 处理嵌套配置对象的转换
        if 'cache' in config_data and isinstance(config_data['cache'], dict):
            config_data['cache'] = CacheConfig.from_dict(config_data['cache'])
        
        if 'processing' in config_data and isinstance(config_data['processing'], dict):
            config_data['processing'] = ProcessingConfig.from_dict(config_data['processing'])
        
        if 'output' in config_data and isinstance(config_data['output'], dict):
            config_data['output'] = OutputConfig.from_dict(config_data['output'])
        
        if 'logging' in config_data and isinstance(config_data['logging'], dict):
            config_data['logging'] = LoggingConfig.from_dict(config_data['logging'])
        
        # 处理传感器配置
        if 'sensors' in config_data:
            sensors = {}
            for sensor_name, sensor_config in config_data['sensors'].items():
                if isinstance(sensor_config, dict):
                    sensors[sensor_name] = SensorConfig.from_dict(sensor_config)
                elif isinstance(sensor_config, SensorConfig):
                    sensors[sensor_name] = sensor_config
                else:
                    raise ValueError(f"Invalid configuration for sensor {sensor_name}")
            config_data['sensors'] = sensors
        
        # 过滤掉未知的配置项
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
        
        # 创建配置实例
        config = cls(**filtered_data)
        return config
    
    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """保存配置到YAML文件"""
        self.validate()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]) -> "WorkflowConfig":
        """从YAML文件加载配置"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        config = cls.from_dict(config_dict)
        config.validate()
        return config
    
    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """保存配置到JSON文件"""
        self.validate()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> "WorkflowConfig":
        """从JSON文件加载配置"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        config.validate()
        return config
    
    def get_enabled_sensors(self) -> Dict[str, SensorConfig]:
        """获取已启用的传感器配置"""
        return {name: config for name, config in self.sensors.items() if config.enabled}
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"WorkflowConfig(data_roots={len(self.data_roots)} roots, cache_dir={self.cache.cache_dir}, sensors={list(self.sensors.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()

# 预定义的配置模板
DEFAULT_CONFIG = WorkflowConfig()

DEVELOPMENT_CONFIG = WorkflowConfig(
    data_roots=["./sample_data"],
    cache=CacheConfig(
        enabled=True,
        cache_dir="./dev_cache",
        validate_cache=True
    ),
    processing=ProcessingConfig(
        use_parallel=True,
        batch_size=500,
        progress_bar=True,
        memory_threshold_mb=1000
    ),
    logging=LoggingConfig(
        level=LogLevel.DEBUG,
        file="./dev.log"
    )
)

PRODUCTION_CONFIG = WorkflowConfig(
    data_roots=["/data/sensors", "/mnt/backup/sensor_data"],
    cache=CacheConfig(
        enabled=True,
        cache_dir="/var/cache/timeseries",
        validate_cache=True,
        compression=True
    ),
    processing=ProcessingConfig(
        use_parallel=True,
        batch_size=2000,
        max_workers=8,
        memory_threshold_mb=4000,
        progress_bar=False
    ),
    output=OutputConfig(
        auto_generate_report=True,
        report_dir="/var/reports/timeseries",
        export_formats=["json", "csv"]
    ),
    logging=LoggingConfig(
        level=LogLevel.INFO,
        file="/var/log/timeseries_workflow.log"
    )
)

def load_config(config_path: Optional[Union[str, Path]] = None) -> WorkflowConfig:
    """
    加载配置，按以下顺序查找：
    1. 指定的配置文件
    2. 环境变量 WORKFLOW_CONFIG
    3. 当前目录下的 config.yaml
    4. 默认配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        WorkflowConfig实例
    """
    if config_path:
        config_path = Path(config_path)
        if config_path.suffix.lower() == '.json':
            return WorkflowConfig.load_from_json(config_path)
        else:
            return WorkflowConfig.load_from_yaml(config_path)
    
    # 检查环境变量
    env_config = os.environ.get("WORKFLOW_CONFIG")
    if env_config:
        env_config_path = Path(env_config)
        if env_config_path.exists():
            if env_config_path.suffix.lower() == '.json':
                return WorkflowConfig.load_from_json(env_config_path)
            else:
                return WorkflowConfig.load_from_yaml(env_config_path)
    
    # 检查当前目录
    current_config_yaml = Path("config.yaml")
    if current_config_yaml.exists():
        return WorkflowConfig.load_from_yaml(current_config_yaml)
    
    current_config_json = Path("config.json")
    if current_config_json.exists():
        return WorkflowConfig.load_from_json(current_config_json)
    
    # 返回默认配置
    return DEFAULT_CONFIG

def create_sample_config(output_path: Union[str, Path] = "sample_config.yaml") -> None:
    """
    创建示例配置文件
    
    Args:
        output_path: 输出文件路径
    """
    config = DEVELOPMENT_CONFIG
    config.save_to_yaml(output_path)
    logging.info(f"Sample config created at {output_path}")

# 测试函数
def test_config_loading():
    """测试配置加载功能"""
    print("Testing configuration loading...")
    
    # 创建测试配置
    test_config = WorkflowConfig(
        data_roots=["/test/path"],
        cache=CacheConfig(cache_dir="/test/cache"),
        processing=ProcessingConfig(batch_size=500),
        sensors={
            "TEST": SensorConfig(pattern=r".*\.TEST$", description="Test Sensor")
        }
    )
    
    # 测试序列化和反序列化
    config_dict = test_config.to_dict()
    print("Serialized config:", config_dict)
    
    reloaded_config = WorkflowConfig.from_dict(config_dict)
    print("Reloaded config type check:")
    print(f"  cache type: {type(reloaded_config.cache)}")
    print(f"  processing type: {type(reloaded_config.processing)}")
    print(f"  output type: {type(reloaded_config.output)}")
    print(f"  logging type: {type(reloaded_config.logging)}")
    print(f"  sensors type: {type(reloaded_config.sensors)}")
    
    # 验证配置
    try:
        reloaded_config.validate()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")

if __name__ == "__main__":
    # 运行测试
    test_config_loading()
    
    # 尝试加载您的配置
    try:
        config_path = r"F:\Research\Vibration Characteristics In Cable Vibration\config\time_series_db.yaml"
        config = load_config(config_path)
        print(f"\n✓ Configuration loaded successfully from {config_path}")
        print(f"Config details: {config}")
        
        # 检查配置对象类型
        print(f"\nConfiguration object types:")
        print(f"  cache: {type(config.cache)} - {config.cache}")
        print(f"  processing: {type(config.processing)} - {config.processing}")
        print(f"  output: {type(config.output)} - {config.output}")
        print(f"  logging: {type(config.logging)} - {config.logging}")
        print(f"  sensors: {type(config.sensors)} - {len(config.sensors)} sensors")
        
    except Exception as e:
        print(f"\n✗ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()