import yaml
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, ClassVar, Any, Union
import pandas as pd
from pathlib import Path


class SensorConfig:
    """传感器配置管理类，根据传感器ID的命名规律自动分组，支持YAML持久化和文件路径映射"""
    
    # 类变量，用于存储版本信息
    _CONFIG_VERSION: ClassVar[str] = "2.0"  # 版本升级，支持文件映射
    
    def __init__(self, sensor_ids: Optional[List[str]] = None):
        """
        初始化传感器配置
        params:
            sensor_ids: List[str]，传感器ID列表，如果为None，则需要通过load_from_yaml加载
        """
        self.sensor_ids = sensor_ids or []
        self.core_sensors = []
        self.vibration_groups = {}
        self.base_time_columns = ['Index', 'timestamp', 'month', 'day', 'hour', 'minute']
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": self._CONFIG_VERSION,
            "description": "Auto-generated sensor configuration"
        }
        
        # 新增：文件路径映射相关属性
        self.file_mappings = {}  # 组名 -> 文件路径映射
        self.base_save_dir = "./sensor_database"  # 默认保存目录
        self.interval_nums = 60  # 默认区间切分数量
        self.time_range = {}  # 存储时间范围信息
        
        # 如果提供了sensor_ids，自动分析和分组
        if sensor_ids:
            self._analyze_and_group_sensors()
    
    def _analyze_and_group_sensors(self):
        """分析传感器ID命名规律，自动分组"""
        if not self.sensor_ids:
            return
        
        # 第一步：分离核心传感器
        # 根据命名规律，UAN标识的传感器为核心传感器
        self.core_sensors = [sid for sid in self.sensor_ids if 'UAN' in sid]
        
        # 第二步：处理振动传感器
        vibration_sensors = [sid for sid in self.sensor_ids if 'VIC' in sid]
        
        # 振动传感器命名规律：ST-VIC-C{location}-{group}{level}-0{axis}
        for sensor_id in vibration_sensors:
            parts = sensor_id.split('-')
            if len(parts) >= 5:
                # 提取位置 (C34, C18)
                location = parts[2]  # C34 or C18
                # 提取组和层级 (101, 102, 201, 202, etc.)
                group_level = parts[3]  # 101, 102, etc.
                # 提取轴 (01, 02)
                axis = parts[4]  # 01, 02
                
                # 创建分组键，例如 "C34_floor1", "C18_floor2"
                floor_num = group_level[0]  # 1, 2, 3, 4
                group_type = group_level[1:3]  # 01, 02
                group_key = f"{location}_floor{floor_num}_group{group_type}"
                
                # 添加到分组
                if group_key not in self.vibration_groups:
                    self.vibration_groups[group_key] = []
                self.vibration_groups[group_key].append(sensor_id)
    
    def get_sensor_id_config(self) -> Dict[str, List[str]]:
        """
        获取传感器ID配置字典，适合传入build_database函数
        return:
            Dict，格式为 {
                "core": [核心时间列 + 核心传感器],
                "vibration_C34_floor1": [传感器ID列表],
                "vibration_C18_floor1": [传感器ID列表],
                ...
            }
        """
        config = {}
        
        # 核心组：包含基础时间列和核心传感器
        config["core"] = self.base_time_columns.copy()
        config["core"].extend(self.core_sensors)
        
        # 振动传感器组
        for group_name, sensors in self.vibration_groups.items():
            # 为每个振动组添加前缀，区分于核心组
            config_key = f"vibration_{group_name}"
            config[config_key] = sensors.copy()
        
        return config
    
    def update_file_mappings(self, file_mappings: Dict[str, str], base_dir: Optional[str] = None):
        """
        更新文件路径映射
        params:
            file_mappings: Dict[str, str]，组名到文件路径的映射
            base_dir: Optional[str]，基础目录，用于计算相对路径
        """
        if base_dir is None:
            base_dir = self.base_save_dir
        
        # 转换为绝对路径并存储相对路径
        self.file_mappings = {}
        base_path = Path(base_dir).resolve()
        
        for group_name, file_path in file_mappings.items():
            if file_path:
                abs_path = Path(file_path).resolve()
                # 计算相对于base_dir的相对路径
                try:
                    rel_path = abs_path.relative_to(base_path)
                    self.file_mappings[group_name] = str(rel_path)
                except ValueError:
                    # 如果不在同一文件系统，保留绝对路径
                    self.file_mappings[group_name] = str(abs_path)
        
        # 更新基础目录
        self.base_save_dir = str(base_path)
        
        # 更新元数据
        self.metadata.update({
            "file_mappings_updated": datetime.now().isoformat(),
            "file_mappings_count": len(self.file_mappings)
        })
    
    def get_file_path(self, group_name: str) -> Optional[str]:
        """
        获取指定组的文件路径（返回绝对路径）
        params:
            group_name: str，组名
        return:
            Optional[str]，文件的绝对路径，如果不存在则返回None
        """
        if group_name in self.file_mappings:
            file_path = self.file_mappings[group_name]
            base_path = Path(self.base_save_dir).resolve()
            
            # 检查是否是绝对路径
            if Path(file_path).is_absolute():
                return str(Path(file_path).resolve())
            else:
                # 否则，相对于base_save_dir
                return str((base_path / file_path).resolve())
        
        return None
    
    def validate_file_mappings(self) -> Dict[str, bool]:
        """
        验证所有映射文件是否存在
        return:
            Dict[str, bool]，组名到文件存在状态的映射
        """
        validation_results = {}
        
        for group_name, rel_path in self.file_mappings.items():
            abs_path = self.get_file_path(group_name)
            validation_results[group_name] = os.path.exists(abs_path) if abs_path else False
        
        return validation_results
    
    def get_group_summary(self) -> Dict:
        """获取传感器分组摘要信息，包括文件映射信息"""
        summary = {
            "total_sensors": len(self.sensor_ids),
            "core_sensors": len(self.core_sensors),
            "vibration_groups": len(self.vibration_groups),
            "vibration_sensors": sum(len(sensors) for sensors in self.vibration_groups.values()),
            "file_mappings_count": len(self.file_mappings),
            "base_save_dir": self.base_save_dir,
            "interval_nums": self.interval_nums,
            "metadata": self.metadata
        }
        
        # 详细分组信息
        group_details = {}
        for group_name, sensors in self.vibration_groups.items():
            group_key = f"vibration_{group_name}"
            file_path = self.get_file_path(group_key)
            group_details[group_name] = {
                "count": len(sensors),
                "sensors": sensors,
                "file_path": file_path if file_path else None,
                "file_exists": os.path.exists(file_path) if file_path else False
            }
        
        summary["group_details"] = group_details
        
        # 核心组信息
        core_file_path = self.get_file_path("core")
        summary["core_info"] = {
            "sensors": self.core_sensors,
            "file_path": core_file_path if core_file_path else None,
            "file_exists": os.path.exists(core_file_path) if core_file_path else False
        }
        
        # 时间范围信息
        if self.time_range:
            summary["time_range"] = self.time_range
        
        return summary
    
    def save_to_yaml(self, file_path: str, include_metadata: bool = True, include_file_mappings: bool = True) -> bool:
        """
        将传感器配置保存为YAML文件
        params:
            file_path: str，YAML文件路径
            include_metadata: bool，是否包含元数据
            include_file_mappings: bool，是否包含文件映射
        return:
            bool，是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 构建YAML结构
            yaml_data = {
                "version": self._CONFIG_VERSION,
                "base_time_columns": self.base_time_columns,
                "core_sensors": self.core_sensors,
                "vibration_groups": self.vibration_groups,
                "all_sensor_ids": self.sensor_ids,
                "base_save_dir": self.base_save_dir,
                "interval_nums": self.interval_nums
            }
            
            if include_file_mappings and self.file_mappings:
                yaml_data["file_mappings"] = self.file_mappings
            
            if self.time_range:
                yaml_data["time_range"] = self.time_range
            
            if include_metadata:
                yaml_data["metadata"] = {
                    "created_at": self.metadata.get("created_at", datetime.now().isoformat()),
                    "updated_at": datetime.now().isoformat(),
                    "description": self.metadata.get("description", "Sensor configuration"),
                    "source": self.metadata.get("source", "auto-generated")
                }
            
            # 写入YAML文件
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            print(f"传感器配置已保存至: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存YAML文件失败: {str(e)}")
            return False
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'SensorConfig':
        """
        从YAML文件加载传感器配置，包括文件映射信息
        params:
            file_path: str，YAML文件路径
        return:
            SensorConfig，加载后的配置对象
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # 检查版本兼容性
            version = yaml_data.get("version", "1.0")
            if version < "2.0":
                print(f"注意: 加载旧版本配置文件 {version}，将进行兼容性转换")
            
            # 创建新实例
            config = cls()
            
            # 加载基础数据
            config.base_time_columns = yaml_data.get("base_time_columns", ['Index', 'timestamp', 'month', 'day', 'hour', 'minute'])
            config.core_sensors = yaml_data.get("core_sensors", [])
            config.vibration_groups = yaml_data.get("vibration_groups", {})
            config.sensor_ids = yaml_data.get("all_sensor_ids", [])
            config.base_save_dir = yaml_data.get("base_save_dir", "./sensor_database")
            config.interval_nums = yaml_data.get("interval_nums", 60)
            
            # 加载文件映射
            config.file_mappings = yaml_data.get("file_mappings", {})
            
            # 加载时间范围
            config.time_range = yaml_data.get("time_range", {})
            
            # 加载元数据
            if "metadata" in yaml_data:
                config.metadata = yaml_data["metadata"]
                config.metadata["loaded_at"] = datetime.now().isoformat()
            else:
                config.metadata = {
                    "loaded_at": datetime.now().isoformat(),
                    "version": version,
                    "source": "loaded-from-yaml"
                }
            
            print(f"传感器配置已从 {file_path} 加载，包含 {len(config.file_mappings)} 个文件映射")
            return config
            
        except Exception as e:
            print(f"加载YAML文件失败: {str(e)}")
            # 返回一个空配置
            return cls()
    
    def update_from_dataframe(self, df: pd.DataFrame):
        """
        从DataFrame更新传感器配置
        params:
            df: pd.DataFrame，包含传感器数据的DataFrame
        """
        # 从DataFrame列名中提取传感器ID
        sensor_columns = [col for col in df.columns 
                         if col not in self.base_time_columns 
                         and not col.startswith('segment_')
                         and not col.startswith('Unnamed')]
        
        # 提取纯传感器ID（去除后缀）
        sensor_ids = []
        for col in sensor_columns:
            # 尝试从列名中提取传感器ID
            # 例如：ST-UAN-G04-001-01-data -> ST-UAN-G04-001-01
            parts = col.split('-')
            if len(parts) >= 5 and (parts[1] == 'UAN' or parts[1] == 'VIC'):
                # 重新组合前5部分作为传感器ID
                sensor_id = '-'.join(parts[:5])
                if sensor_id not in sensor_ids:
                    sensor_ids.append(sensor_id)
        
        self.sensor_ids = sensor_ids
        self._analyze_and_group_sensors()
        
        # 更新元数据
        self.metadata.update({
            "updated_at": datetime.now().isoformat(),
            "source": "updated-from-dataframe",
            "dataframe_shape": str(df.shape)
        })
    
    def update_time_range(self, start_time: Union[str, datetime], end_time: Union[str, datetime], total_records: int):
        """
        更新时间范围信息
        params:
            start_time: 开始时间
            end_time: 结束时间
            total_records: 总记录数
        """
        if isinstance(start_time, datetime):
            start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(end_time, datetime):
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        self.time_range = {
            "start_time": start_time,
            "end_time": end_time,
            "total_records": total_records
        }
    
    def __str__(self):
        """返回配置的字符串表示，包含文件映射信息"""
        summary = self.get_group_summary()
        result = f"传感器配置摘要 (v{summary['metadata'].get('version', '1.0')}):\n"
        result += f"  总传感器数: {summary['total_sensors']}\n"
        result += f"  核心传感器: {summary['core_sensors']}\n"
        result += f"  振动传感器组: {summary['vibration_groups']}\n"
        result += f"  振动传感器总数: {summary['vibration_sensors']}\n"
        result += f"  文件映射数量: {summary['file_mappings_count']}\n"
        result += f"  基础保存目录: {summary['base_save_dir']}\n"
        result += f"  区间切分数量: {summary['interval_nums']}\n\n"
        
        # 时间范围
        if self.time_range:
            result += f"时间范围:\n"
            result += f"  开始时间: {self.time_range.get('start_time', 'N/A')}\n"
            result += f"  结束时间: {self.time_range.get('end_time', 'N/A')}\n"
            result += f"  总记录数: {self.time_range.get('total_records', 'N/A')}\n\n"
        
        result += "振动传感器分组详情:\n"
        for i, (group_name, details) in enumerate(summary["group_details"].items(), 1):
            result += f"  组 {i}: {group_name} ({details['count']} 个传感器)\n"
            # 只显示前3个传感器，避免输出过长
            sensors_preview = details['sensors'][:3]
            if len(details['sensors']) > 3:
                sensors_preview.append(f"...等{len(details['sensors'])}个")
            result += f"    传感器: {', '.join(sensors_preview)}\n"
            
            # 显示文件映射信息
            group_key = f"vibration_{group_name}"
            file_path = details.get('file_path')
            file_exists = details.get('file_exists', False)
            if file_path:
                result += f"    数据文件: {file_path} {'[存在]' if file_exists else '[不存在]'}\n"
        
        # 核心组文件信息
        core_info = summary.get("core_info", {})
        core_file_path = core_info.get('file_path')
        core_file_exists = core_info.get('file_exists', False)
        if core_file_path:
            result += f"\n核心数据文件: {core_file_path} {'[存在]' if core_file_exists else '[不存在]'}\n"
        
        return result