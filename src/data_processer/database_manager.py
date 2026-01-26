from typing import (
    Callable, Dict, List, Optional, Union, Tuple, 
    Any, Iterable, TypeVar, overload, Set
)

import os
import json
import pandas as pd
import numpy as np
import gc
import time
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fnmatch
import inspect
import multiprocessing
import threading

# 配置日志（全局统一配置，可根据需求调整）
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别：INFO（可改为DEBUG显示更多细节）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 创建日志实例

# （注：需确保当前类已包含 __init__ 中初始化的 config、core_columns、base_dir 等属性）


class ChunkManager:
    def __init__(self, local_dir: str, config_filename: str = "config.json"):
        """
        初始化分块管理器
        
        params:
            local_dir: str，数据库及其配置文件所在的基础目录
            config_filename: str，配置文件名，默认为"config.json"
        """
        # 1. 标准化基础目录路径（支持相对路径）
        self.base_dir = os.path.abspath(local_dir)
        self.config_path = os.path.join(self.base_dir, config_filename)
        
        # 2. 验证配置文件存在
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        # 3. 加载JSON配置文件
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件解析错误: {str(e)}")
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")
        
        # 4. 验证必要配置项（移除core块强制校验）
        required_sections = ['file_mappings']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必需的 '{section}' 配置项")
        
        # 6. 存储计算结果
        if "calculation_results" not in self.config:
            self.config["calculation_results"] = []

        # 验证传感器组配置（确保sensor_groups和file_mappings块名一致）
        self._validate_sensor_group_mappings()

        # 新增：收集所有传感器ID（去重），用于列名前缀匹配
        self.all_sensor_ids = self._collect_all_sensor_ids()
        logger.info(f"共收集到 {len(self.all_sensor_ids)} 个唯一传感器ID")

        # 新增：可配置的列名分隔符（用户可根据实际列名调整）
        self.column_separators = self.config.get(
            "column_separators", 
            ["-", "_", ".", ":"]  # 常见分隔符，覆盖大部分场景
        )

        # 新增：定义系统列（无需分配函数的列，默认包含timestamp，用户可通过config扩展）
        self.base_time_columns = self.config["metadata"].get(
            "base_time_columns",
            ["timestamp",
                "month",
                "day",
                "hour",
                "minute"])  # 默认系统列，支持用户在config根节点添加（如 ["timestamp", "id"]）
        
        # 确保base_time_columns是列表且元素为字符串
        if not isinstance(self.base_time_columns, list) or not all(isinstance(col, str) for col in self.base_time_columns):
            raise TypeError("config中的 base_time_columns 必须是字符串列表")
        self.base_time_columns = [col.strip() for col in self.base_time_columns if col.strip()]
        logger.info(f"系统列（无需分配函数）：{self.base_time_columns}")

        # 关键修复1：多进程共享数据（原普通字典子进程无法访问）
        # 多进程共享管理器（仅初始化一次）
        self.manager = multiprocessing.Manager()
        # 共享数据块字典（多进程可访问）
        self.chunks = self.manager.dict()
        # 共享缓存（dict 模拟 set，多进程可访问）
        self._computed_cache = self.manager.dict()
        # 关键修正：跨进程安全锁（替换 threading.Lock）
        self.chunk_lock = self.manager.Lock()  # 跨进程安全锁
        self.cache_lock = self.manager.Lock()  # 跨进程安全锁
        

    def _validate_sensor_group_mappings(self):
        """辅助校验：sensor_groups中的块名是否都在file_mappings中存在"""
        group_blocks = set(self.config.get("sensor_groups", {}).keys())
        file_blocks = set(self.config.get("file_mappings", {}).keys())
        missing_blocks = group_blocks - file_blocks - {"core"}  # core不属于传感器组
        if missing_blocks:
            warnings.warn(f"传感器组中以下块名未在file_mappings中找到：{missing_blocks}")

    def _collect_all_sensor_ids(self) -> List[str]:
        """辅助函数：从 sensor_groups 中收集所有唯一传感器ID"""
        all_ids = set()
        sensor_groups = self.config.get("sensor_groups", {})
        for group_sensor_configs in sensor_groups.values():
            for config in group_sensor_configs:
                if isinstance(config, str):
                    # 纯传感器ID字符串
                    sensor_id = config.strip()
                    if sensor_id:
                        all_ids.add(sensor_id)
                elif isinstance(config, dict):
                    # 字典配置（含 sensor_id 字段）
                    sensor_id = config.get("sensor_id", "").strip()
                    if sensor_id:
                        all_ids.add(sensor_id)
        logger.debug(f"收集到的传感器ID列表：{sorted(list(all_ids))}")  # 调试信息用DEBUG级别
        return sorted(list(all_ids))  # 排序后返回，便于调试

    def _column_exists_in_chunks(self, col_name: str) -> bool:
        """
        检查列是否存在于任何数据块中（基于传感器ID前缀匹配，无文件读取）
        匹配规则：列名以传感器ID为前缀，且前缀后是分隔符或列名结束（避免部分匹配）
        """
        col_name = str(col_name).strip()
        if not col_name:
            return False

        # 1. 检查是否是 timestamp 列（所有块必含，直接返回True）
        if col_name == "timestamp":
            return True

        # 2. 传感器列校验：前缀匹配传感器ID（原有核心逻辑不变）
        for sensor_id in self.all_sensor_ids:
            if not sensor_id:
                continue
            if not col_name.startswith(sensor_id):
                continue
            
            prefix_len = len(sensor_id)
            if len(col_name) == prefix_len:
                return True
            elif len(col_name) > prefix_len and col_name[prefix_len] in self.column_separators:
                return True

        # 3. 未匹配到任何传感器ID或timestamp列
        logger.debug(f"列 '{col_name}' 未匹配到任何传感器ID或系统列")
        return False

    def _parse_simple_mapping(
        self,
        target_col: Union[str, List[str]],
        funcs: Union[List[Callable], List[List[Callable]], List[Tuple[str, Callable]]]
    ) -> Dict[str, Dict[str, List[Callable]]]:
        """
        私有函数：处理「简单模式」- 快速批量映射（target_col + funcs）
        自动过滤系统列，不分配函数
        """
        # 步骤1：解析target_col，获取完整列列表、函数映射列列表、原始target→列映射
        parsed_cols, parsed_cols_for_mapping, target_to_cols = self._resolve_target_columns(target_col)
        
        # 校验：函数映射列不能为空（过滤系统列后无有效列）
        if not parsed_cols_for_mapping:
            raise ValueError("简单模式：过滤系统列后未找到有效业务列，无法分配函数")
        
        # 统计有效业务列数量
        logger.info(f"简单模式：共解析到 {len(parsed_cols_for_mapping)} 个有效业务列")

        # 步骤2：标准化funcs（基于原始target→完整列映射，精准广播）- 原有逻辑不变
        standardized_funcs_full = self._standardize_funcs_for_simple_mode(
            funcs=funcs,
            target_to_cols=target_to_cols,
            parsed_cols=parsed_cols
        )
        
        # 步骤3：过滤系统列对应的函数，只保留业务列的函数映射 - 原有逻辑不变
        col_to_funcs = dict(zip(parsed_cols, standardized_funcs_full))
        valid_pairs = [
            (col, func_list) 
            for col, func_list in col_to_funcs.items() 
            if col in parsed_cols_for_mapping and func_list
        ]
        
        if not valid_pairs:
            warnings.warn("简单模式：所有业务列对应的函数列表均为空，返回空映射")
            return {}
        # 统计有效业务列-函数映射数
        logger.info(f"简单模式：有效业务列-函数映射数：{len(valid_pairs)}")

        # 步骤4：关联列→块，构建最终映射
        mapping = {}
        allow_func_override = self.config.get("allow_func_override", False)

        for col, func_list in valid_pairs:
            block_info = self._find_column_belonging_block(col)
            if not block_info:
                warnings.warn(f"简单模式：列 '{col}' 未找到所属块，跳过该列-函数映射")
                continue
            
            block_name = block_info["block_name"]
            block_type = block_info["block_type"]

            # 初始化块映射
            if block_name not in mapping:
                mapping[block_name] = {}

            # 处理函数冲突
            if col in mapping[block_name]:
                old_funcs = mapping[block_name][col]
                if allow_func_override:
                    logger.warning(
                        f"简单模式：块 '{block_name}' 业务列 '{col}' 函数冲突，覆盖原有函数："
                        f"{[f.__name__ for f in old_funcs]} → {[f.__name__ for f in func_list]}"
                    )
                    mapping[block_name][col] = func_list
                else:
                    logger.warning(
                        f"简单模式：块 '{block_name}' 业务列 '{col}' 函数冲突，保留原有函数（未覆盖）"
                    )
                continue
            
            # 新增列-函数映射
            func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in func_list]
            logger.info(
                f"简单模式：块 '{block_name}'（{block_type}）→ 业务列 '{col}' → 函数：{func_names}"
            )
            mapping[block_name][col] = func_list

        # 最终映射统计
        logger.info(f"简单模式：映射构建完成，共涉及 {len(mapping)} 个块")
        for block_name, col_funcs in mapping.items():
            logger.debug(f"  - 块 '{block_name}'：共 {len(col_funcs)} 个列-函数映射")

        return mapping
    
    def _parse_batch_mapping(
        self,
        func_mapping: Dict[str, List[Union[Callable, Tuple[str, Callable]]]]
    ) -> Dict[str, Dict[str, List[Callable]]]:
        """私有函数：处理「批量模式」- 字典格式批量映射（键：块标识/列名/通配符，值：函数列表）"""
        mapping = {}
        # 读取函数覆盖配置（与简单模式保持一致）
        allow_func_override = self.config.get("allow_func_override", False)
        
        # 遍历字典的每个「键-函数列表」对
        for key, raw_funcs in func_mapping.items():
            if not raw_funcs:
                warnings.warn(f"批量模式：键 '{key}' 对应的函数列表为空，跳过")
                continue
            
            # 步骤1：解析key为具体列列表（支持block:xxx、通配符、列名）
            parsed_cols = self._resolve_batch_key_to_columns(key)
            if not parsed_cols:
                warnings.warn(f"批量模式：键 '{key}' 未解析到有效列，跳过")
                continue
            
            # 新增：过滤系统列（仅保留业务列进行函数映射）
            valid_cols = [col for col in parsed_cols if col not in self.base_time_columns]
            if not valid_cols:
                warnings.warn(f"批量模式：键 '{key}' 解析的列均为系统列，无需分配函数，跳过")
                continue
            
            # 步骤2：提取纯函数（处理别名元组）
            func_list = self._extract_pure_functions(raw_funcs)
            if not func_list:
                warnings.warn(f"批量模式：键 '{key}' 未提取到有效函数，跳过")
                continue
            
            # 步骤3：分配到列→块，构建映射（补充冲突处理和日志）
            for col in valid_cols:
                block_info = self._find_column_belonging_block(col)
                if not block_info:
                    warnings.warn(f"批量模式：列 '{col}' 未找到所属块，跳过")
                    continue
                
                block_name = block_info["block_name"]
                block_type = block_info["block_type"]  # 新增：获取块类型
                
                # 初始化块映射
                if block_name not in mapping:
                    mapping[block_name] = {}
                
                # 新增：处理函数冲突
                if col in mapping[block_name]:
                    old_funcs = mapping[block_name][col]
                    if allow_func_override:
                        old_func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in old_funcs]
                        new_func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in func_list]
                        logger.warning(
                            f"批量模式：块 '{block_name}' 业务列 '{col}' 函数冲突，覆盖原有函数："
                            f"{old_func_names} → {new_func_names}"
                        )
                        mapping[block_name][col] = func_list
                    else:
                        logger.warning(
                            f"批量模式：块 '{block_name}' 业务列 '{col}' 函数冲突，保留原有函数（未覆盖）"
                        )
                    continue
                
                # 新增：打印映射日志（与简单模式格式一致）
                func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in func_list]
                logger.info(
                    f"批量模式：块 '{block_name}'（{block_type}）→ 业务列 '{col}' → 函数：{func_names}"
                )
                
                # 追加函数（去重）
                if col not in mapping[block_name]:
                    mapping[block_name][col] = []
                for func in func_list:
                    if func not in mapping[block_name][col]:
                        mapping[block_name][col].append(func)
        
        # 新增：映射结果统计日志（便于调试）
        logger.info(f"批量模式：映射构建完成，共涉及 {len(mapping)} 个块")
        for block_name, col_funcs in mapping.items():
            logger.debug(f"  - 块 '{block_name}'：共 {len(col_funcs)} 个列-函数映射")
        
        return mapping

    def _parse_full_mapping(
        self,
        func_mapping: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Callable]]]:
        """私有函数：处理「完整模式」- 列表字典格式精细映射（支持块/列/函数的精准配置）"""
        mapping = {}
        # 读取函数覆盖配置（与其他模式保持一致）
        allow_func_override = self.config.get("allow_func_override", False)
        
        # 遍历每个精细映射项
        for idx, item in enumerate(func_mapping):
            # 校验必填项
            if "funcs" not in item or not item["funcs"]:
                raise ValueError(f"完整模式：第{idx}个映射项缺少有效 'funcs' 参数")
            
            # 提取配置项（block可选、columns可选默认'all'、alias可选）
            block_name = item.get("block")
            columns = item.get("columns", "all")
            raw_funcs = item["funcs"]
            alias = item.get("alias", f"mapping_item_{idx}")
            
            # 步骤1：解析列列表（根据是否指定block适配逻辑）
            parsed_cols = self._resolve_full_mode_columns(columns, block_name, alias)
            if not parsed_cols:
                warnings.warn(f"完整模式：{alias} 未解析到有效列，跳过")
                continue
            
            # 新增：过滤系统列（仅保留业务列进行函数映射）
            valid_cols = [col for col in parsed_cols if col not in self.base_time_columns]
            if not valid_cols:
                warnings.warn(f"完整模式：{alias} 解析的列均为系统列，无需分配函数，跳过")
                continue
            
            # 步骤2：提取纯函数 + 新增校验（无有效函数则跳过）
            func_list = self._extract_pure_functions(raw_funcs)
            if not func_list:
                warnings.warn(f"完整模式：{alias} 未提取到有效函数，跳过")
                continue
            
            # 步骤3：分配到列→块，构建映射（补充冲突处理、日志输出）
            for col in valid_cols:
                # 若未指定block，自动查找列所属块
                final_block = block_name
                block_type = "sensor_group"  # 默认块类型
                if final_block is None:
                    block_info = self._find_column_belonging_block(col)
                    if not block_info:
                        warnings.warn(f"完整模式：{alias} 列 '{col}' 未找到所属块，跳过")
                        continue
                    final_block = block_info["block_name"]
                    block_type = block_info["block_type"]  # 从块信息中获取真实类型
                else:
                    # 若指定了block，补充获取块类型（用于日志）
                    block_info = self._find_column_belonging_block(col)
                    if block_info:
                        block_type = block_info["block_type"]
                
                # 初始化块映射
                if final_block not in mapping:
                    mapping[final_block] = {}
                
                # 新增：处理函数冲突（与其他模式逻辑一致）
                if col in mapping[final_block]:
                    old_funcs = mapping[final_block][col]
                    if allow_func_override:
                        old_func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in old_funcs]
                        new_func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in func_list]
                        logger.warning(
                            f"完整模式：{alias} - 块 '{final_block}' 业务列 '{col}' 函数冲突，覆盖原有函数："
                            f"{old_func_names} → {new_func_names}"
                        )
                        mapping[final_block][col] = func_list
                    else:
                        logger.warning(
                            f"完整模式：{alias} - 块 '{final_block}' 业务列 '{col}' 函数冲突，保留原有函数（未覆盖）"
                        )
                    continue
                
                # 新增：打印详细映射日志（含别名、块类型，便于调试）
                func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in func_list]
                logger.info(
                    f"完整模式：{alias} - 块 '{final_block}'（{block_type}）→ 业务列 '{col}' → 函数：{func_names}"
                )
                
                # 追加函数（去重，保持原有逻辑）
                if col not in mapping[final_block]:
                    mapping[final_block][col] = []
                for func in func_list:
                    if func not in mapping[final_block][col]:
                        mapping[final_block][col].append(func)
        
        # 新增：映射结果统计日志（与其他模式格式统一）
        logger.info(f"完整模式：映射构建完成，共涉及 {len(mapping)} 个块")
        for block_name, col_funcs in mapping.items():
            logger.debug(f"  - 块 '{block_name}'：共 {len(col_funcs)} 个列-函数映射")
        
        return mapping

    # ------------------------------ 以下为辅助函数（支撑核心逻辑）------------------------------
    def _judge_mapping_mode(
        self,
        target_col: Any,
        funcs: Any,
        func_mapping: Any
    ) -> str:
        """辅助函数：判断映射模式（simple/batch/full）并校验输入合法性"""
        # 模式优先级：func_mapping > target_col+funcs
        if func_mapping is not None:
            if isinstance(func_mapping, dict):
                return "batch"
            elif isinstance(func_mapping, list):
                return "full"
            else:
                raise TypeError("func_mapping 必须是字典（批量模式）或列表（完整模式）")
        else:
            if target_col is None or funcs is None:
                raise ValueError("简单模式必须同时传入 target_col 和 funcs 参数")
            return "simple"

    def _resolve_target_columns(
        self,
        target: Union[str, List[str]]
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        辅助函数：自动识别输入类型，解析简单模式的target_col
        返回：(parsed_cols, parsed_cols_for_mapping, target_to_cols)
            - parsed_cols：包含系统列的完整列列表（数据加载用）
            - parsed_cols_for_mapping：过滤系统列后的列列表（函数映射用）
            - target_to_cols：字典，key=原始target项，value=该target解析出的完整列列表（含系统列）
        """
        targets = [target] if isinstance(target, str) else target
        if not isinstance(targets, list):
            raise TypeError(f"target_col 必须是字符串或列表，当前类型：{type(target)}")
        
        all_available_cols = self._get_all_available_columns()
        all_available_blocks = list(self.config.get("file_mappings", {}).keys())
        target_to_cols = {}  # 原始target → 完整列列表（含系统列）
        raw_parsed_cols = []  # 未去重的完整列列表（含系统列）
        
        for t in targets:
            t = str(t).strip()
            if not t:
                target_to_cols[t] = []
                continue
            
            # 按优先级解析当前target项（保留系统列）
            current_cols = []
            if t == "all":
                current_cols = all_available_cols.copy()
            elif "*" in t or "?" in t:
                current_cols = [col for col in all_available_cols if fnmatch.fnmatch(col, t)]
            else:
                # 块名（带前缀/直接输入）或列名
                block_name = None
                if t.startswith("block:"):
                    block_name = t[6:].strip()
                elif t in all_available_blocks:
                    block_name = t
                
                if block_name:
                    try:
                        current_cols = self._get_block_columns(block_name)
                    except Exception as e:
                        warnings.warn(f"块 '{block_name}' 解析失败：{str(e)}，跳过")
                else:
                    # 列名（保留系统列，后续过滤）
                    if self._column_exists_in_chunks(t):
                        current_cols = [t]
                        logger.debug(f"解析单个列名 '{t}'：识别为有效列，加入完整列列表")
                    else:
                        warnings.warn(f"输入 '{t}' 无效（非块名/传感器列/通配符），跳过")
            
            # 记录映射关系，收集未去重列
            target_to_cols[t] = current_cols
            raw_parsed_cols.extend(current_cols)
        
        # 去重并保留插入顺序（完整列列表，含系统列）
        parsed_cols = list(dict.fromkeys(raw_parsed_cols))
        # 过滤系统列（用于函数映射的列列表）
        parsed_cols_for_mapping = [col for col in parsed_cols if col not in self.base_time_columns]
        
        # 解析结果汇总
        logger.info(f"\n解析结果汇总：")
        logger.info(f"  - 完整列数（含系统列）：{len(parsed_cols)} 列")
        logger.info(f"  - 函数映射列数（过滤系统列）：{len(parsed_cols_for_mapping)} 列")
        if self.base_time_columns:
            logger.info(f"  - 已过滤系统列：{self.base_time_columns}")
        
        return parsed_cols, parsed_cols_for_mapping, target_to_cols
    
    def _standardize_funcs_for_simple_mode(
        self,
        funcs: Union[List[Callable], List[List[Callable]], List[Tuple[str, Callable]]],
        target_to_cols: Dict[str, List[str]],  # 原始target→解析列的映射
        parsed_cols: List[str]  # 最终去重列列表（确保顺序一致）
    ) -> List[List[Callable]]:
        """
        辅助函数：标准化简单模式的funcs，实现精准广播
        逻辑：funcs与原始target一一对应，target解析出N列则函数广播N次，最终与parsed_cols顺序一致
        """
        # 提取纯函数（复用原逻辑，处理别名元组）
        def extract_pure(func_items):
            pure_funcs = []
            for item in func_items:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and callable(item[1]):
                    pure_funcs.append(item[1])
                elif callable(item):
                    pure_funcs.append(item)
                else:
                    raise TypeError(f"函数项 '{item}' 无效（需为可调用对象或(别名, 函数)元组）")
            return pure_funcs
        
        # 校验：funcs长度必须与原始target数量一致（一一对应）
        raw_target_count = len([t for t in target_to_cols.keys() if t.strip()])
        if len(funcs) != raw_target_count:
            raise ValueError(
                f"funcs长度（{len(funcs)}）必须与有效原始target数量（{raw_target_count}）一致\n"
                f"原始target列表：{list(target_to_cols.keys())}\n"
                f"提示：每个target项（列名/块名/通配符）需对应一个函数或函数列表"
            )
        
        # 步骤1：构建「原始target→标准化函数列表」的映射
        target_to_funcs = {}
        target_index = 0  # 遍历funcs的索引
        for raw_target, cols in target_to_cols.items():
            raw_target = raw_target.strip()
            if not raw_target:
                continue
            
            func_item = funcs[target_index]
            target_index += 1
            
            # 标准化当前target对应的函数
            if isinstance(func_item, list) and (callable(func_item[0]) or isinstance(func_item[0], tuple)):
                # 单个函数列表（广播到该target的所有列）
                pure_func_list = extract_pure(func_item)
                target_to_funcs[raw_target] = pure_func_list
            elif isinstance(func_item, list) and isinstance(func_item[0], list):
                # 列表的列表（每个列单独指定函数，需校验长度）
                if len(func_item) != len(cols):
                    raise ValueError(
                        f"target '{raw_target}' 解析出 {len(cols)} 列，但对应的函数列表长度为 {len(func_item)}\n"
                        f"提示：若target是块名/通配符（多列），请传入单个函数列表（自动广播）；若需单独指定，函数列表长度需与列数一致"
                    )
                # 提取每个子列表的纯函数
                target_to_funcs[raw_target] = [extract_pure(sub_funcs) for sub_funcs in func_item]
            else:
                raise TypeError(
                    f"funcs[{target_index-1}]（对应target '{raw_target}'）格式无效\n"
                    f"支持格式：1. 函数列表（如 [np.mean, np.max]）；2. 列表的列表（如 [[np.mean], [np.max]]）"
                )
        
        # 步骤2：基于parsed_cols的顺序，组装最终的函数列表（精准广播）
        standardized_funcs = []
        # 为了去重列的函数冲突：记录列已分配的函数（保留第一个分配的函数）
        col_func_cache = {}
        
        for col in parsed_cols:
            if col in col_func_cache:
                # 去重列：直接复用已分配的函数
                standardized_funcs.append(col_func_cache[col])
                continue
            
            # 查找该列对应的原始target，获取函数
            for raw_target, cols in target_to_cols.items():
                if col in cols:
                    target_func = target_to_funcs[raw_target]
                    if isinstance(target_func, list) and not isinstance(target_func[0], list):
                        # 单个函数列表（广播场景）
                        func_list = target_func.copy()
                    else:
                        # 列表的列表（单个列场景）
                        col_index = cols.index(col)
                        func_list = target_func[col_index].copy()
                    
                    # 缓存并添加到结果
                    col_func_cache[col] = func_list
                    standardized_funcs.append(func_list)
                    break
        
        logger.debug(f"简单模式函数标准化完成，共生成 {len(standardized_funcs)} 个函数列表（与列数对应）")
        return standardized_funcs

    def _resolve_batch_key_to_columns(self, key: str) -> List[str]:
        """辅助函数：批量模式自动识别key类型（块名/列名/通配符，无需block:前缀）"""
        all_available_cols = self._get_all_available_columns()
        all_available_blocks = list(self.config.get("file_mappings", {}).keys())
        parsed_cols = []
        
        key = str(key).strip()
        if not key:
            return parsed_cols
        
        # 1. 通配符优先
        if "*" in key or "?" in key:
            parsed_cols = [col for col in all_available_cols if fnmatch.fnmatch(col, key)]
            logger.debug(f"批量模式：通配符 '{key}' 匹配到 {len(parsed_cols)} 列")
            return list(set(parsed_cols))
        
        # 2. 块名（支持直接输入/block:前缀）
        block_name = None
        if key.startswith("block:"):
            block_name = key[6:].strip()
        elif key in all_available_blocks:
            block_name = key
        
        if block_name:
            try:
                parsed_cols = self._get_block_columns(block_name)
                logger.debug(f"批量模式：块 '{block_name}' 解析到 {len(parsed_cols)} 列")
                return list(set(parsed_cols))
            except Exception as e:
                warnings.warn(f"批量模式：块 '{block_name}' 解析失败：{str(e)}，跳过")
                return []
        
        # 3. 列名
        if self._column_exists_in_chunks(key):
            parsed_cols = [key]
            logger.debug(f"批量模式：key '{key}' 识别为列名，加入解析列表")
        
        return parsed_cols

    def _resolve_full_mode_columns(
        self,
        columns: Union[str, List[str]],
        block_name: Optional[str],
        alias: str
    ) -> List[str]:
        """辅助函数：解析完整模式的columns配置"""
        all_available_cols = self._get_all_available_columns()
        parsed_cols = []
        
        # 统一转为列表处理
        cols = [columns] if isinstance(columns, str) else columns
        if not isinstance(cols, list):
            raise TypeError(f"完整模式 {alias}：columns 必须是字符串或列表")
        
        for col in cols:
            col = str(col).strip()
            if not col:
                continue
            
            if col == "all":
                # columns='all' 必须配合 block 参数
                if block_name is None:
                    raise ValueError(f"完整模式 {alias}：columns='all' 必须指定 block 参数")
                parsed_cols.extend(self._get_block_columns(block_name))
                logger.debug(f"完整模式 {alias}：block '{block_name}' + columns='all' 解析到 {len(parsed_cols)} 列")
            elif "*" in col or "?" in col:
                # 通配符匹配
                matched = [c for c in all_available_cols if fnmatch.fnmatch(c, col)]
                parsed_cols.extend(matched)
                logger.debug(f"完整模式 {alias}：通配符 '{col}' 匹配到 {len(matched)} 列")
            else:
                # 列名 → 校验存在性
                if self._column_exists_in_chunks(col):
                    parsed_cols.append(col)
                    logger.debug(f"完整模式 {alias}：列 '{col}' 存在，加入解析列表")
                else:
                    warnings.warn(f"完整模式 {alias}：列 '{col}' 不存在，跳过")
        
        return list(set(parsed_cols))

    def _find_column_belonging_block(self, col_name: str) -> Optional[Dict[str, str]]:
        """
        查找列所属的块信息（基于传感器ID前缀匹配，无文件读取，兼容所有列名格式）
        匹配规则：列名以传感器ID为前缀，且前缀后是分隔符或列名结束 → 找到包含该传感器ID的块
        """
        col_name = str(col_name).strip()
        if not col_name:
            return None

        # 1. 匹配timestamp列（所有块都含该列，返回第一个传感器块）
        if col_name == "timestamp":
            sensor_blocks = [b for b in self.config["file_mappings"].keys()]
            if sensor_blocks:
                first_block = sensor_blocks[0]
                logger.debug(f"列 '{col_name}' 为系统时间列，分配到第一个传感器块 '{first_block}'")
                return {
                    "block_name": first_block,
                    "block_type": "sensor_group",
                    "file_path": self.config["file_mappings"][first_block]
                }
            return None

        # 2. 传感器列：通过前缀匹配找到对应的传感器ID
        matched_sensor_id = None
        for sensor_id in self.all_sensor_ids:
            if not sensor_id:
                continue
            
            if col_name == sensor_id:
                matched_sensor_id = sensor_id
                break
            
            prefix_len = len(sensor_id)
            if len(col_name) > prefix_len and col_name.startswith(sensor_id) and col_name[prefix_len] in self.column_separators:
                matched_sensor_id = sensor_id
                break

        if not matched_sensor_id:
            logger.debug(f"列 '{col_name}' 未匹配到任何传感器ID")
            return None

        # 3. 遍历sensor_groups，找到包含该传感器ID的块
        sensor_groups = self.config.get("sensor_groups", {})
        for block_name, group_sensor_configs in sensor_groups.items():
            if block_name not in self.config["file_mappings"]:
                continue
            
            for config in group_sensor_configs:
                if isinstance(config, str) and config.strip() == matched_sensor_id:
                    logger.debug(f"列 '{col_name}'（传感器ID：{matched_sensor_id}）分配到块 '{block_name}'")
                    return {
                        "block_name": block_name,
                        "block_type": "sensor_group",
                        "file_path": self.config["file_mappings"][block_name]
                    }
                elif isinstance(config, dict) and config.get("sensor_id", "").strip() == matched_sensor_id:
                    logger.debug(f"列 '{col_name}'（传感器ID：{matched_sensor_id}）分配到块 '{block_name}'")
                    return {
                        "block_name": block_name,
                        "block_type": "sensor_group",
                        "file_path": self.config["file_mappings"][block_name]
                    }

        # 4. 找到传感器ID，但未找到对应的块（配置异常）
        warnings.warn(f"传感器ID '{matched_sensor_id}' 未关联到任何有效块，列 '{col_name}' 无法分配块")
        return None

    def _get_all_available_columns(self) -> List[str]:
        """辅助函数：获取所有可用列（所有传感器块列）"""
        all_cols = set()
        for block_name in self.config["file_mappings"]:
            all_cols.update(self._get_block_columns(block_name))
        logger.debug(f"所有可用列（去重后）：{sorted(list(all_cols))}")
        return list(all_cols)

    def _get_block_columns(self, block_name: str) -> List[str]:
        """
        辅助函数：灵活推导指定块的所有列名（无文件读取，兼容单一列/多列传感器）
        逻辑优先级：
        1. 传感器ID配置了具体列名 → 直接使用
        2. 无特殊配置 → 默认单一列（仅传感器ID）
        """
        # 1. 从sensor_groups获取该块的所有传感器配置（支持字符串ID或字典配置）
        sensor_groups = self.config.get("sensor_groups", {})
        if block_name not in sensor_groups:
            raise KeyError(f"配置文件 sensor_groups 中未找到块 '{block_name}' 的传感器配置")
        
        sensor_configs = sensor_groups[block_name]
        if not isinstance(sensor_configs, list) or len(sensor_configs) == 0:
            raise ValueError(f"块 '{block_name}' 的传感器配置为空或格式错误")

        block_cols = []
        for config in sensor_configs:
            # 处理配置类型：字符串（仅传感器ID）或字典（详细配置）
            if isinstance(config, str):
                sensor_id = config.strip()
                # 无特殊配置 → 默认单一列（仅传感器ID）
                block_cols.append(sensor_id)
            elif isinstance(config, dict):
                sensor_id = config.get("sensor_id")
                if not sensor_id:
                    warnings.warn(f"块 '{block_name}' 中存在无sensor_id的无效配置，跳过")
                    continue
                
                # 优先级1：使用配置中指定的具体列名
                if "columns" in config and isinstance(config["columns"], list):
                    specified_cols = [col.strip() for col in config["columns"] if col.strip()]
                    if specified_cols:
                        block_cols.extend(specified_cols)
                        continue
                
                # 优先级2：默认单一列（仅传感器ID）
                block_cols.append(sensor_id)
            else:
                warnings.warn(f"块 '{block_name}' 中存在无效的传感器配置类型（{type(config)}），跳过")
                continue

        unique_cols = list(dict.fromkeys(block_cols))  # 保留插入顺序的去重
        logger.debug(f"块 '{block_name}' 推导列名：{unique_cols}")
        return unique_cols
    
    def _derive_cols_from_core(self, sensor_id: str) -> List[str]:
        """辅助函数：已移除核心列逻辑，不再支持从核心列推导列名，返回空列表"""
        # 核心列相关逻辑已移除，该函数仅保留接口兼容性
        logger.debug(f"_derive_cols_from_core：核心列逻辑已移除，传感器ID '{sensor_id}' 无推导列")
        return []
    
    def _extract_pure_functions(
        self,
        raw_funcs: List[Union[Callable, Tuple[str, Callable]]]
    ) -> List[Callable]:
        """辅助函数：从原始函数列表中提取纯可调用函数（过滤别名）"""
        pure_funcs = []
        for item in raw_funcs:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and callable(item[1]):
                pure_funcs.append(item[1])
            elif callable(item):
                pure_funcs.append(item)
            else:
                raise TypeError(f"函数项 '{item}' 无效（需为可调用对象或(别名, 函数)元组）")
        func_names = [f.__name__ if hasattr(f, '__name__') else 'lambda_func' for f in pure_funcs]
        logger.debug(f"提取纯函数列表：{func_names}")
        return pure_funcs

    def _validate_mapping_result(self, mapping: Dict[str, Dict[str, List[Callable]]]) -> None:
        """辅助函数：校验最终映射结果的有效性"""
        if not mapping:
            raise ValueError("解析后的函数映射为空，请检查输入配置")
        
        for block_name, col_funcs in mapping.items():
            # 校验块存在性
            if block_name not in self.config["file_mappings"]:
                raise ValueError(f"映射中包含不存在的块：{block_name}")
            # 校验列-函数有效性
            for col, funcs in col_funcs.items():
                if not funcs:
                    raise ValueError(f"块 '{block_name}' 的列 '{col}' 未关联任何有效函数")
                if not all(callable(f) for f in funcs):
                    raise TypeError(f"块 '{block_name}' 的列 '{col}' 包含非可调用函数项")
        logger.info(f"函数映射校验通过，共 {len(mapping)} 个块，{sum(len(v) for v in mapping.values())} 个列-函数映射")

    def _compute_single_block(
        self,
        block_name: str,
        col_funcs: Dict[str, List[Callable]],
        force_recompute: bool = False,
        error_handling: str = "skip",
        fillna_strategy: Dict[str, Any] = {'method': 'ignore', 'value': np.nan},
        padding_strategy: Dict[str, Any] = {'method': 'nan', 'value': np.nan},
        computed_cache: Union[set, "multiprocessing.managers.SetProxy"] = None  # 新增：进程安全缓存
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:  # 修改：返回 (块结果, 计算后的数据块)
        """
        私有方法：执行单个数据块的运算（适配并发，支持进程安全缓存）
        新增参数说明：
            computed_cache: 进程安全的计算缓存（并发时传入，单进程时用实例自带的 _computed_cache）
        返回值说明：
            Tuple[块结果摘要, 计算后的数据块] - 主进程统一合并结果，避免多进程冲突
        """
        # 1. 初始化配置与基础数据（复用原有逻辑）
        init_result = self._init_compute_config(
            block_name=block_name,
            col_funcs=col_funcs,
            fillna_strategy=fillna_strategy,
            padding_strategy=padding_strategy
        )
        block_result, chunk_df, original_cols, is_main_process = init_result

        # 2. 并发适配：使用进程安全缓存（单进程时默认用实例的 _computed_cache）
        if computed_cache is None:
            computed_cache = self._computed_cache

        # 3. 遍历传感器ID → 匹配列 → 逐列计算（核心逻辑不变，仅替换缓存为传入的 computed_cache）
        for sensor_id, func_list in col_funcs.items():
            sensor_result = self._process_single_sensor(
                sensor_id=sensor_id,
                func_list=func_list,
                original_cols=original_cols,
                is_main_process=is_main_process
            )
            if not sensor_result["valid"]:
                self._update_block_result(
                    block_result=block_result,
                    success=False,
                    sensor_id=sensor_id,
                    real_col_name=None,
                    func_name="all",
                    error=sensor_result["error"]
                )
                continue

            matched_cols = sensor_result["matched_cols"]
            block_result["total_matched_cols"] += len(matched_cols)
            if is_main_process:
                logger.info(f"\n块 '{block_name}' 传感器ID '{sensor_id}' 匹配到 {len(matched_cols)} 列：{matched_cols}")

            for real_col_name in matched_cols:
                # 关键修改：将 computed_cache 传入 _process_single_column（替代实例的 self._computed_cache）
                col_compute_result = self._process_single_column(
                    block_name=block_name,
                    real_col_name=real_col_name,
                    func_list=func_list,
                    chunk_df=chunk_df,
                    force_recompute=force_recompute,
                    error_handling=error_handling,
                    fillna_strategy=fillna_strategy,
                    padding_strategy=padding_strategy,
                    is_main_process=is_main_process,
                    computed_cache=computed_cache  # 新增：传递进程安全缓存
                )
                self._update_block_result(
                    block_result=block_result,
                    success=col_compute_result["success"],
                    sensor_id=sensor_id,
                    real_col_name=real_col_name,
                    func_name=col_compute_result.get("func_name"),
                    error=col_compute_result.get("error"),
                    func_info=col_compute_result.get("func_info")
                )

        # 4. 关键修改：不再直接修改实例的 chunks 和 config（主进程统一合并）
        if is_main_process:
            self._print_block_summary(block_result, is_main_process)

        # 返回块结果和计算后的数据块（主进程接收后合并）
        return block_result, chunk_df

    @staticmethod
    def _preprocess_column_data(col_data: pd.Series) -> pd.Series:
        """静态方法：基础数据清理"""
        def clean_element(x):
            if isinstance(x, str):
                return x.strip()
            return x

        col_data = col_data.apply(clean_element)
        logger.debug(f"列数据预处理完成，非空值数量：{col_data.notna().sum()}")
        return col_data

    @staticmethod
    def _safe_execute_element(x: Any, func: Callable, func_kwargs: Dict[str, Any]) -> Any:
        """静态方法：安全执行元素级函数"""
        try:
            kwargs = func_kwargs if isinstance(func_kwargs, dict) else {}
            return func(x, **kwargs)
        except Exception as e:
            is_main_process = multiprocessing.current_process().name == "MainProcess"
            if is_main_process:
                func_name = func.__name__ if hasattr(func, "__name__") else "lambda_func"
                logger.warning(
                    f"函数 {func_name} 执行失败 → "
                    f"元素值：{x}，元素类型：{type(x)}，"
                    f"错误类型：{type(e).__name__}，错误信息：{str(e)[:150]}"
                )
            return x

    def _execute_single_function(
        self,
        func: Callable,
        valid_data: pd.Series,
        original_index: pd.Index,
        real_col_name: str,
        padding_strategy: Dict[str, Any],
        error_handling: str,
        is_main_process: bool
    ) -> Dict[str, Any]:
        """执行单个函数的元素级计算（确保实际执行并返回有效结果）"""
        result = {"success": False, "func_name": "", "new_col_name": "", "result_padded": None, "error": ""}
        func_name = func.__name__ if hasattr(func, "__name__") else "lambda_func"
        new_col_name = f"{real_col_name}_{func_name}"
        result["func_name"] = func_name
        result["new_col_name"] = new_col_name

        try:
            exec_result = valid_data.apply(func)  # 元素级执行
            # 校验执行结果（避免空结果）
            if exec_result.empty or exec_result.notna().sum() == 0:
                result["error"] = f"元素级计算无有效结果（输入数据非空但输出全缺失）"
                return result

            # 按策略补全索引（确保与原始数据一致）
            result_padded = self._pad_result(exec_result, original_index, padding_strategy)
            result["result_padded"] = result_padded
            result["success"] = True
            if is_main_process:
                logger.info(f"函数 {func_name} 执行成功：新列 {new_col_name}，有效结果数 {result_padded.notna().sum()}")

        except Exception as e:
            # 异常处理策略
            if error_handling == "retry":
                try:
                    if is_main_process:
                        logger.info(f"函数 {func_name} 执行失败，重试一次...")
                    exec_result = valid_data.apply(func)
                    result_padded = self._pad_result(exec_result, original_index, padding_strategy)
                    result["result_padded"] = result_padded
                    result["success"] = True
                    if is_main_process:
                        logger.info(f"函数 {func_name} 重试成功：新列 {new_col_name}")
                except Exception as retry_e:
                    result["error"] = f"重试后仍失败：{str(retry_e)}"
                    logger.error(f"函数 {func_name} 重试失败：{str(retry_e)}")
            else:
                result["error"] = str(e)
                logger.error(f"函数 {func_name} 执行失败：{str(e)}")

        return result

    # -------------------------- 保留的模块化辅助方法（未修改） --------------------------
    def _init_compute_config(
        self,
        block_name: str,
        col_funcs: Dict[str, List[Callable]],
        fillna_strategy: Dict[str, Any],
        padding_strategy: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], pd.DataFrame, List[str], bool]:
        fillna_strategy = fillna_strategy or {"method": "ignore", "value": np.nan}
        if fillna_strategy["method"] not in ["drop", "fill", "ignore"]:
            raise ValueError(f"无效的缺失值处理方法：{fillna_strategy['method']}，支持 drop/fill/ignore")

        padding_strategy = padding_strategy or {"method": "nan", "value": np.nan}
        if padding_strategy["method"] not in ["nan", "value", "backfill", "forwardfill"]:
            raise ValueError(f"无效的Padding方法：{padding_strategy['method']}，支持 nan/value/backfill/forwardfill")

        is_main_process = multiprocessing.current_process().name == "MainProcess"
        if is_main_process:
            logger.info(f"初始化块 '{block_name}' 计算配置：缺失值处理={fillna_strategy['method']}，Padding策略={padding_strategy['method']}")

        block_result = {
            "block_name": block_name,
            "compute_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sensor_ids": len(col_funcs),
            "total_matched_cols": 0,
            "success_cols": 0,
            "failed_cols": 0,
            "details": {"success": [], "failed": []}
        }

        if block_name not in self.chunks:
            raise KeyError(f"数据块 '{block_name}' 未加载，请先调用 load_chunk('{block_name}')")
        chunk_df = self.chunks[block_name].copy()
        original_cols = chunk_df.columns.tolist()

        self._computed_cache = getattr(self, "_computed_cache", set())

        return block_result, chunk_df, original_cols, is_main_process

    def _process_single_sensor(
        self,
        sensor_id: str,
        func_list: List[Callable],
        original_cols: List[str],
        is_main_process: bool
    ) -> Dict[str, Any]:
        sensor_id = sensor_id.strip()
        result = {"valid": False, "matched_cols": [], "error": ""}

        if not sensor_id:
            result["error"] = "传感器ID为空"
            return result

        if not func_list or not all(callable(f) for f in func_list):
            result["error"] = "函数列表为空或包含非可调用对象"
            return result

        matched_cols = self._match_sensor_related_columns(sensor_id, original_cols)
        if not matched_cols:
            if is_main_process:
                warnings.warn(f"块 '{list(self.chunks.keys())[0]}' 传感器ID '{sensor_id}' 未匹配到任何列")
            result["error"] = "未匹配到符合规则的列（传感器ID+分隔符+后缀）"
            return result

        result["valid"] = True
        result["matched_cols"] = matched_cols
        return result

    @staticmethod
    def _handle_missing_values(col_data: pd.Series, fillna_strategy: Dict[str, Any]) -> pd.Series:
        """静态方法：处理缺失值"""
        method = fillna_strategy.get("method", "ignore")
        value = fillna_strategy.get("value", np.nan)

        if method == "drop":
            result = col_data.dropna()
        elif method == "fill":
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"填充值必须是标量（当前：{type(value)}）")
            result = col_data.fillna(value)
        else:
            result = col_data.fillna(np.nan)
        
        logger.debug(f"缺失值处理（{method}）：非空行数 {result.notna().sum()}")
        return result
        
    @staticmethod
    def _pad_result(exec_result: pd.Series, original_index: pd.Index, padding_strategy: Dict[str, Any]) -> pd.Series:
        """静态方法：补全结果索引（优化后：允许布尔值结果，减弱过滤逻辑）"""
        # 1. 索引对齐（保持原有逻辑）
        padded = exec_result.reindex(original_index)
        pad_method = padding_strategy.get("method", "nan")
        pad_value = padding_strategy.get("value", np.nan)

        # 辅助函数：判断是否为布尔类型（保持原有定义）
        def is_boolean_type(x):
            return isinstance(x, (bool, np.bool_))

        # 辅助函数：判断是否为数组类类型（保持原有定义）
        def is_array_like_type(x):
            if isinstance(x, str):  # 字符串排除（避免误判为数组类）
                return False
            return isinstance(x, (np.ndarray, list))

        # -------------------------- 核心优化1：减弱数组类布尔值过滤 --------------------------
        def light_process_array_like(x):
            """仅处理空数组，保留数组中的布尔值（不再转为np.nan）"""
            if isinstance(x, np.ndarray):
                arr = x.copy()
                # 仅保留空数组处理（原有逻辑），移除布尔值替换
                if len(arr) == 0 or arr.size == 0:
                    return arr
                # 【删除】原逻辑：布尔类型数组转为float64+nan
                # 【删除】原逻辑：数组中布尔值替换为nan
                return arr  # 直接返回原数组（保留所有类型，包括布尔值）
            
            elif isinstance(x, list):
                # 仅保留空列表处理（原有逻辑），移除布尔值替换
                if len(x) == 0:
                    return x
                # 【修改】不再替换列表中的布尔值，直接返回原列表
                return x.copy()
            
            return x  # 非数组类直接返回

        # -------------------------- 核心优化2：保留单个布尔值（移除强制转为nan的逻辑） --------------------------
        # 【删除】原逻辑：padded = padded.apply(lambda x: np.nan if (not is_array_like_type(x) and is_boolean_type(x)) else x)
        # 直接保留所有非数组类布尔值（True/False）

        # 数组类数据轻量处理（仅处理空数组）
        padded = padded.apply(lambda x: light_process_array_like(x) if is_array_like_type(x) else x)

        # -------------------------- 核心优化3：允许布尔值填充 --------------------------
        if pad_method == "fill":
            # 【删除】原逻辑：禁止布尔值填充的异常判断
            # 允许使用布尔值填充（如 pad_value=True/False）
            padded = padded.fillna(pad_value)

        return padded

    def _update_block_result(
        self,
        block_result: Dict[str, Any],
        success: bool,
        sensor_id: str,
        real_col_name: str,
        func_name: str,
        error: str = "",
        func_info: Tuple[List[str], List[str], str] = None
    ) -> None:
        if success:
            func_names, new_cols, status = func_info
            block_result["success_cols"] += 1
            block_result["details"]["success"].append({
                "sensor_id": sensor_id,
                "real_col_name": real_col_name,
                "funcs": func_names,
                "new_cols": new_cols,
                "status": status
            })
            logger.info(f"列 '{real_col_name}' 计算成功：新增列 {new_cols}，状态：{status}")
        else:
            block_result["failed_cols"] += 1
            block_result["details"]["failed"].append({
                "sensor_id": sensor_id,
                "real_col_name": real_col_name,
                "func_name": func_name,
                "error": error
            })
            logger.error(f"列 '{real_col_name}' 计算失败：函数 '{func_name}'，错误：{error[:200]}")

    def _print_block_summary(self, block_result: Dict[str, Any], is_main_process: bool) -> None:
        if not is_main_process:
            return
        logger.info(f"\n【块 '{block_result['block_name']}' 计算完成】")
        logger.info(f"  - 输入传感器ID数：{block_result['total_sensor_ids']}")
        logger.info(f"  - 匹配真实列数：{block_result['total_matched_cols']}")
        logger.info(f"  - 成功列数：{block_result['success_cols']}")
        logger.info(f"  - 失败列数：{block_result['failed_cols']}")
        new_cols_count = len([item for sublist in [s['new_cols'] for s in block_result['details']['success']] for item in sublist])
        logger.info(f"  - 新增列数：{new_cols_count}")

    def _match_sensor_related_columns(self, sensor_id: str, all_cols: List[str]) -> List[str]:
        matched_cols = []
        sensor_id_len = len(sensor_id)
        for col in all_cols:
            col = str(col).strip()
            if not col.startswith(sensor_id):
                continue
            if len(col) == sensor_id_len or col[sensor_id_len] in self.column_separators:
                matched_cols.append(col)
        unique_matched = sorted(list(dict.fromkeys(matched_cols)))
        logger.debug(f"传感器ID '{sensor_id}' 匹配到列：{unique_matched}")
        return unique_matched

    @staticmethod
    def _check_function_compatibility(func: Callable, func_name: str, col_name: str):
        """静态方法：函数兼容性校验"""
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if len(params) == 0:
            logger.warning(f"函数 '{func_name}' 无参数，建议按 'def {func_name}(x: float, **kwargs) -> float' 定义")
        else:
            first_param = params[0]
            allowed_annotations = [float, int, Any, inspect.Parameter.empty]
            if first_param.annotation not in allowed_annotations:
                logger.warning(
                    f"函数 '{func_name}' 第一个参数注解为 {first_param.annotation}（建议float/int）\n"
                    f"列 '{col_name}' 中不符合类型的元素将返回原数据"
                )

    def _process_single_column(
        self,
        block_name: str,
        real_col_name: str,
        func_list: List[Callable],
        chunk_df: pd.DataFrame,
        force_recompute: bool,
        error_handling: str,
        fillna_strategy: Dict[str, Any],
        padding_strategy: Dict[str, Any],
        computed_cache: Union[set, "multiprocessing.managers.SetProxy"],
        is_main_process: bool
    ) -> Dict[str, Any]:
        result = {
            "success": False,
            "func_info": None,
            "func_name": None,
            "error": "",
            "new_cols_written": []
        }
        computed_key = (block_name, real_col_name)
        base_log = f"[块:{block_name} 列:{real_col_name}]"

        # -------------------------- 1. 基础校验 --------------------------
        if real_col_name not in chunk_df.columns:
            result["error"] = f"{base_log} 原始列不存在"
            logger.error(result["error"])
            return result

        raw_col_data = chunk_df[real_col_name]
        original_index = raw_col_data.index
        if is_main_process:
            logger.info(f"{base_log} 原始数据：行数={len(raw_col_data)} 非空行数={raw_col_data.notna().sum()}")

        # -------------------------- 2. 缓存逻辑 --------------------------
        cache_hit = False
        if not force_recompute and isinstance(computed_cache, (set, "multiprocessing.managers.SetProxy")):
            cache_hit = computed_key in computed_cache

        if cache_hit:
            if is_main_process:
                logger.info(f"{base_log} 已缓存，跳过计算")
            func_names = [f.__name__ if hasattr(f, "__name__") else "lambda_func" for f in func_list]
            new_cols = [f"{real_col_name}_{name}" for name in func_names]
            result["success"] = True
            result["func_info"] = (func_names, new_cols, "skipped（已缓存）")
            result["new_cols_written"] = new_cols
            return result

        # -------------------------- 3. 关键修复：解析列表列为标量列 --------------------------
        # 检查是否为列表列（每行是列表）
        is_list_column = raw_col_data.apply(lambda x: isinstance(x, list)).any()
        if is_list_column:
            if is_main_process:
                logger.info(f"{base_log} 检测到列表列，开始解析为标量列...")
            # 解析列表列为标量列（每行单个值）
            col_data = self._parse_list_column(raw_col_data)
            if is_main_process:
                logger.info(f"{base_log} 列表列解析完成：非空标量行数={col_data.notna().sum()}")
        else:
            # 普通标量列，直接预处理
            col_data = self._preprocess_column_data(raw_col_data)

        # 校验解析后的数据有效性
        if col_data.empty or col_data.notna().sum() == 0:
            result["error"] = f"{base_log} 解析后无有效标量数据"
            logger.error(result["error"])
            return result

        # -------------------------- 4. 缺失值处理（对标量列）--------------------------
        valid_data = self._handle_missing_values(col_data, fillna_strategy)
        if valid_data.notna().sum() == 0:
            result["error"] = f"{base_log} 缺失值处理后无有效数据"
            logger.error(result["error"])
            return result

        # -------------------------- 5. 元素级计算（确保每行返回标量）--------------------------
        col_success = True
        func_names = []
        new_cols = []
        written_new_cols = []

        for idx, func in enumerate(func_list):
            func_name = func.__name__ if hasattr(func, "__name__") else f"lambda_{idx}"
            new_col_name = f"{real_col_name}_{func_name}"
            func_log = f"{base_log} 函数[{func_name}]"

            if is_main_process:
                logger.info(f"{func_log} 开始元素级计算...")

            try:
                # 关键修复：对每个标量执行函数（确保元素级）
                # 用 np.vectorize 确保函数支持向量化计算（比 apply 更高效）
                vectorized_func = np.vectorize(func, otypes=[np.float64])
                exec_result = pd.Series(
                    vectorized_func(valid_data.values),  # 对标量数组执行函数
                    index=valid_data.index,
                    name=new_col_name
                )

                # 校验计算结果（必须是标量）
                if exec_result.apply(lambda x: isinstance(x, (int, float, np.number))).sum() == 0:
                    raise ValueError("计算结果非标量（可能是列表/None）")

                # -------------------------- 6. 修复 Padding 补全（避免 None）--------------------------
                # 补全到原始索引，确保无 None
                result_padded = self._pad_result(exec_result, original_index, padding_strategy)

                # 替换 Padding 后的 None 为策略值（避免 None 残留）
                pad_method = padding_strategy.get("method", "nan")
                pad_value = padding_strategy.get("value", np.nan)
                if pad_method == "fill":
                    result_padded = result_padded.fillna(pad_value)
                else:
                    result_padded = result_padded.fillna(np.nan)  # 统一用 np.nan 替代 None

                # 写入新列（确保是标量列）
                chunk_df[new_col_name] = result_padded
                written_new_cols.append(new_col_name)
                func_names.append(func_name)
                new_cols.append(new_col_name)

                if is_main_process:
                    logger.info(f"{func_log} 计算成功：新列={new_col_name} 有效标量数={result_padded.notna().sum()}")

            except Exception as e:
                col_success = False
                result["error"] = f"{func_log} 执行失败：{str(e)}"
                result["func_name"] = func_name
                if is_main_process:
                    logger.error(f"{result['error']}")
                break

        # -------------------------- 7. 缓存与结果清理 --------------------------
        if col_success:
            try:
                computed_cache.add(computed_key)
                if is_main_process:
                    logger.debug(f"{base_log} 缓存添加成功：{computed_key}")
            except Exception as e:
                if is_main_process:
                    logger.error(f"{base_log} 缓存添加失败：{str(e)}")

            result["success"] = True
            result["func_info"] = (func_names, new_cols, "computed（成功）")
            result["new_cols_written"] = written_new_cols
        else:
            # 清理部分写入的新列
            for new_col in written_new_cols:
                if new_col in chunk_df.columns:
                    del chunk_df[new_col]
                    if is_main_process:
                        logger.info(f"{base_log} 清理失败列：{new_col}")

        return result
    
    def _process_single_column_concurrent(
        self,
        block_name: str,
        real_col_name: str,
        func_list: List[Callable],
        chunk_df: pd.DataFrame,
        original_index: pd.Index,
        force_recompute: bool,
        error_handling: str,
        fillna_strategy: Dict[str, Any],
        padding_strategy: Dict[str, Any],
        use_cache: bool
    ) -> Dict[str, Any]:
        """
        子进程专用：计算单个列（主进程已预过滤，无需重复匹配传感器）
        返回：列计算结果（成功状态、新列名、错误信息）
        """
        result = {"success": False, "new_cols": [], "func_name": None, "error": ""}
        logger.info(f"开始处理块[{block_name}]列[{real_col_name}]，待执行函数数：{len(func_list)}")
        
        raw_col_data = chunk_df[real_col_name]
        logger.debug(f"块[{block_name}]列[{real_col_name}]原始数据形状：{raw_col_data.shape}，非空值数：{raw_col_data.notna().sum()}")

        # 新增：解析列表列为标量列
        is_list_column = raw_col_data.apply(lambda x: isinstance(x, list)).any()
        if is_list_column:
            logger.info(f"块[{block_name}]列[{real_col_name}]检测为列表列，开始解析为标量列")
            # 解析列表列为标量列（每行单个值）
            col_data = self._parse_list_column(raw_col_data)
            logger.info(f"块[{block_name}]列[{real_col_name}]列表列解析完成，解析后非空值数：{col_data.notna().sum()}")
        else:
            logger.debug(f"块[{block_name}]列[{real_col_name}]为标量列，执行数据预处理")
            # 1. 列数据预处理（简化清理，不修改类型）
            col_data = self._preprocess_column_data(raw_col_data)
            logger.debug(f"块[{block_name}]列[{real_col_name}]预处理完成，预处理后非空值数：{col_data.notna().sum()}")
        
        # 2. 缺失值处理
        logger.info(f"块[{block_name}]列[{real_col_name}]开始缺失值处理，策略：{fillna_strategy['method']}")
        valid_data = self._handle_missing_values(col_data, fillna_strategy)
        valid_count = valid_data.notna().sum()
        logger.info(f"块[{block_name}]列[{real_col_name}]缺失值处理完成，有效数据量：{valid_count}/{len(valid_data)}")

        # 3. 无有效数据则返回失败
        if valid_count == 0:
            error_msg = f"块[{block_name}]列[{real_col_name}]无有效数据（全部为NaN/inf/异常值）"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # 4. 逐函数执行计算
        col_success = True
        new_cols = []
        logger.info(f"块[{block_name}]列[{real_col_name}]开始执行函数计算，共{len(func_list)}个函数")
        
        for idx, func in enumerate(func_list, 1):
            func_name = func.__name__ if hasattr(func, "__name__") else "lambda_func"
            new_col_name = f"{real_col_name}_{func_name}"
            func_kwargs = self.config.get("func_kwargs", {}).get(func_name, {})
            logger.info(f"块[{block_name}]列[{real_col_name}]执行第{idx}/{len(func_list)}个函数：{func_name}，参数：{func_kwargs}")

            try:
                # 校验函数兼容性（仅警告）
                self._check_function_compatibility(func, func_name, real_col_name, is_main_process=False)
                logger.debug(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]兼容性校验通过")
                
                # 逐元素执行
                logger.debug(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]开始逐元素计算")
                result_valid = valid_data.apply(lambda x: self._safe_execute_element(x, func, func_kwargs))
                logger.debug(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]逐元素计算完成，结果非空值数：{result_valid.notna().sum()}")
                
                # Padding补全
                logger.debug(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]开始Padding补全，策略：{padding_strategy['method']}")
                result_padded = self._pad_result(result_valid, original_index, padding_strategy)
                logger.debug(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]Padding补全完成，结果形状：{result_padded.shape}")
                
                # 写入新列
                chunk_df[new_col_name] = result_padded
                new_cols.append(new_col_name)
                logger.info(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]执行成功，新增列：{new_col_name}")
                
            except Exception as e:
                error_msg = str(e)[:200]
                logger.error(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]执行失败：{error_msg}")
                
                # 异常处理：重试
                if error_handling == "retry":
                    logger.info(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]开始重试...")
                    try:
                        result_valid = valid_data.apply(lambda x: self._safe_execute_element(x, func, func_kwargs))
                        result_padded = self._pad_result(result_valid, original_index, padding_strategy)
                        chunk_df[new_col_name] = result_padded
                        new_cols.append(new_col_name)
                        logger.info(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]重试成功，新增列：{new_col_name}")
                    except Exception as retry_e:
                        error_msg = f"重试失败：{str(retry_e)[:200]}"
                        logger.error(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]重试失败：{error_msg}")
                        col_success = False
                        break
                elif error_handling == "abort":
                    logger.critical(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]执行失败，中断计算（error_handling=abort）")
                    raise RuntimeError(f"块 '{block_name}' 列 '{real_col_name}' 函数 '{func_name}' 执行失败，中断计算") from e
                else:  # skip
                    logger.warning(f"块[{block_name}]列[{real_col_name}]函数[{func_name}]执行失败，跳过该函数（error_handling=skip）")
                    col_success = False
                    break

        if col_success:
            result["success"] = True
            result["new_cols"] = new_cols
            logger.info(f"块[{block_name}]列[{real_col_name}]所有函数执行完成，成功新增列：{new_cols}")
        else:
            result["error"] = error_msg
            result["func_name"] = func_name
            logger.error(f"块[{block_name}]列[{real_col_name}]计算失败，失败函数：{func_name}，错误信息：{error_msg}")

        return result

    def _print_concurrent_summary(self, all_block_results: List[Dict[str, Any]]) -> None:
        """打印并发计算全局汇总日志"""
        total_blocks = len(all_block_results)
        total_success_cols = sum([r["success_cols"] for r in all_block_results])
        total_failed_cols = sum([r["failed_cols"] for r in all_block_results])
        total_matched_cols = sum([r["total_matched_cols"] for r in all_block_results])

        logger.info("\n" + "="*50)
        logger.info("=== 并发计算完成 ===")
        logger.info(f"总数据块数：{total_blocks}")
        logger.info(f"总匹配列数：{total_matched_cols}")
        logger.info(f"总成功列数：{total_success_cols}")
        logger.info(f"总失败列数：{total_failed_cols}")
        logger.info("="*50)

    @staticmethod
    def _parse_list_column(col_data: pd.Series) -> pd.Series:
        """
        解析列表列/数组列为标量列：
        - 支持 list 和 np.ndarray 类型元素；
        - 过滤缺失值和非数值类型，返回均值（标量）；
        """
        def parse_list_val(val):
            # 1. 处理缺失值/空值
            if pd.isna(val):
                return np.nan
            
            # 2. 统一将 list/np.array 转为列表，过滤非数值元素
            if isinstance(val, np.ndarray):
                val = val.tolist()  # 数组转列表
            if not isinstance(val, list) or len(val) == 0:
                return np.nan
            
            # 3. 过滤缺失值和非数值类型（确保仅保留 int/float）
            valid_vals = []
            for x in val:
                if pd.isna(x):
                    continue
                # 处理嵌套数组/列表（如果有）
                if isinstance(x, (list, np.ndarray)):
                    # 递归解析嵌套结构，仅保留数值
                    nested_valid = [y for y in (x.tolist() if isinstance(x, np.ndarray) else x) 
                                    if not pd.isna(y) and isinstance(y, (int, float))]
                    valid_vals.extend(nested_valid)
                elif isinstance(x, (int, float)):
                    valid_vals.append(x)
            
            # 4. 计算均值（返回标量），无有效值则返回 NaN
            return np.mean(valid_vals) if valid_vals else np.nan

        return col_data.apply(parse_list_val)

    def _compute_single_block_concurrent(
        self,
        block_name: str,
        col_func_list: List[Tuple[str, List[Callable]]],
        force_recompute: bool,
        error_handling: str,
        fillna_strategy: Dict[str, Any],
        padding_strategy: Dict[str, Any],
        computed_cache: "multiprocessing.managers.DictProxy",
        use_cache: bool,
        local_progress_counter: "multiprocessing.managers.ValueProxy",  # 块内进度
        global_progress_counter: "multiprocessing.managers.ValueProxy"  # 全局进度
    ) -> Tuple[str, Dict[str, Any], pd.DataFrame, Set[Tuple[str, str]]]:
        """子进程专用：计算单个数据块 + 同步块内+全局进度"""
        logger.info(f"子进程开始处理块[{block_name}]，需计算列数：{len(col_func_list)}")
        
        block_result = {
            "block_name": block_name,
            "compute_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sensor_ids": 0,
            "total_matched_cols": len(col_func_list),
            "success_cols": 0,
            "failed_cols": 0,
            "details": {"success": [], "failed": []}
        }

        chunk_df = self.chunks[block_name].copy()
        original_index = chunk_df.index
        new_cached_keys = set()

        for idx, (real_col_name, func_list) in enumerate(col_func_list, 1):
            logger.info(f"子进程处理块[{block_name}]第{idx}/{len(col_func_list)}列：{real_col_name}，函数数：{len(func_list)}")
            
            computed_key = (block_name, real_col_name)
            if use_cache and not force_recompute and computed_key in computed_cache:
                logger.info(f"子进程：块[{block_name}]列[{real_col_name}]已被其他进程计算，跳过")
                # 同时更新块内和全局进度
                local_progress_counter.value += 1
                global_progress_counter.value += 1
                continue

            # 执行列计算
            col_result = self._process_single_column_concurrent(
                block_name=block_name,
                real_col_name=real_col_name,
                func_list=func_list,
                chunk_df=chunk_df,
                original_index=original_index,
                force_recompute=force_recompute,
                error_handling=error_handling,
                fillna_strategy=fillna_strategy,
                padding_strategy=padding_strategy,
                use_cache=use_cache
            )

            # 同时更新块内和全局进度（无论成功失败，均视为完成）
            local_progress_counter.value += 1
            global_progress_counter.value += 1

            # 更新块结果
            if col_result["success"]:
                block_result["success_cols"] += 1
                new_cached_keys.add(computed_key)
                if use_cache:
                    computed_cache[computed_key] = True
                block_result["details"]["success"].append({
                    "sensor_id": "pre_filtered",
                    "real_col_name": real_col_name,
                    "funcs": [f.__name__ if hasattr(f, "__name__") else "lambda_func" for f in func_list],
                    "new_cols": col_result["new_cols"],
                    "status": "computed（成功）"
                })
                logger.info(f"子进程：块[{block_name}]列[{real_col_name}]计算成功，新增列：{col_result['new_cols']}")
            else:
                block_result["failed_cols"] += 1
                block_result["details"]["failed"].append({
                    "sensor_id": "pre_filtered",
                    "real_col_name": real_col_name,
                    "func_name": col_result.get("func_name"),
                    "error": col_result.get("error")
                })
                logger.error(f"子进程：块[{block_name}]列[{real_col_name}]计算失败，失败函数：{col_result.get('func_name')}，错误：{col_result.get('error')}")

        logger.info(f"子进程处理块[{block_name}]完成，成功列数：{block_result['success_cols']}，失败列数：{block_result['failed_cols']}")
        return block_name, block_result, chunk_df, new_cached_keys

    def _parse_function_mapping(
        self,
        target_col: Union[str, List[str]] = None,
        funcs: Union[List[Callable], List[List[Callable]], List[Tuple[str, Callable]]] = None,
        func_mapping: Union[Dict[str, List[Union[Callable, Tuple[str, Callable]]]], List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, List[Callable]]]:
        """
        主函数：统一解析函数映射，自动分发到对应模式，返回标准化「块-列-函数」映射
        
        输出格式：{block_name: {column_name: [func1, func2, ...]}}
        
        参数说明：
            target_col: 简单模式专用 - 目标列标识（'all'/'core'、列列表、块列表(block:xxx)、通配符）
            funcs: 简单模式专用 - 函数列表（支持单个列表、列表的列表、带别名元组(alias, func)）
            func_mapping: 批量/完整模式专用 - 字典（批量）或列表字典（完整）
        """
        # 1. 模式判断与输入校验（二选一：func_mapping 或 target_col+funcs）
        mode = self._judge_mapping_mode(target_col, funcs, func_mapping)
        
        # 2. 分发到对应模式处理
        try:
            if mode == "simple":
                logger.info(f"使用简单模式解析函数映射，目标列：{target_col}，函数数：{len(funcs) if funcs else 0}")
                mapping = self._parse_simple_mapping(target_col, funcs)
            elif mode == "batch":
                logger.info(f"使用批量模式解析函数映射，映射条目数：{len(func_mapping) if func_mapping else 0}")
                mapping = self._parse_batch_mapping(func_mapping)
            else:  # full mode
                logger.info(f"使用完整模式解析函数映射，映射条目数：{len(func_mapping) if func_mapping else 0}")
                mapping = self._parse_full_mapping(func_mapping)
        except Exception as e:
            logger.error(f"函数映射解析失败：{str(e)}")
            raise RuntimeError(f"函数映射解析失败：{str(e)}") from e
        
        # 3. 最终有效性校验（无空函数列表、列/块合法）
        logger.info(f"函数映射解析完成，共{len(mapping)}个块，开始有效性校验")
        self._validate_mapping_result(mapping)
        
        # 日志输出解析结果摘要
        total_cols = 0
        total_funcs = 0
        for block, col_funcs in mapping.items():
            total_cols += len(col_funcs)
            for funcs in col_funcs.values():
                total_funcs += len(funcs)
        logger.info(f"函数映射校验通过，解析结果：{len(mapping)}个块，{total_cols}个列，{total_funcs}个函数")
        
        return mapping

    # ------------------------------ 原有辅助函数（保持不变） ------------------------------
    def _process_single_sensor(self, sensor_id, func_list, original_cols, is_main_process=True):
        """处理单个传感器列匹配（原逻辑不变）"""
        # 示例实现（根据实际需求调整，确保返回 {"valid": bool, "matched_cols": list}）
        matched_cols = [col for col in original_cols if str(sensor_id) in col]
        return {
            "valid": len(matched_cols) > 0,
            "matched_cols": matched_cols
        }

    def _print_concurrent_summary(self, all_block_results):
        """打印计算汇总报告（原逻辑不变）"""
        logger.info("\n" + "="*80)
        logger.info("并发计算汇总报告")
        logger.info("="*80)
        total_success = 0
        total_failed = 0
        total_cols = 0
        for result in all_block_results:
            logger.info(f"块名: {result['block_name']}")
            logger.info(f"  - 匹配列数: {result['total_matched_cols']}")
            logger.info(f"  - 成功列数: {result['success_cols']}")
            logger.info(f"  - 失败列数: {result['failed_cols']}")
            logger.info(f"  - 计算时间: {result['compute_time']}")
            if result['details']['failed']:
                logger.info(f"  - 失败详情: {[f'{f["col_name"]}:{f["error"]}' for f in result['details']['failed'][:3]]}...")
            logger.info("-"*40)
            total_success += result['success_cols']
            total_failed += result['failed_cols']
            total_cols += result['total_matched_cols']
        success_rate = total_success / total_cols if total_cols > 0 else 0.0
        logger.info(f"总汇总: 匹配列数={total_cols}，成功={total_success}，失败={total_failed}，成功率={success_rate:.2%}")
        logger.info("="*80 + "\n")

    @staticmethod
    def _compute_single_block_worker(
        block_name, col_func_list, force_recompute, error_handling,
        fillna_cfg, padding_cfg, chunks, chunk_lock,
        computed_cache, cache_lock, use_cache,
        progress_counter, column_separators, base_time_columns, keep_original_data=False, 
    ):
        """子进程核心工作方法：完整适配本地计算逻辑"""
        try:
            # 安全获取数据块
            with chunk_lock:
                chunk_df = chunks[block_name].copy()
            
            # 记录初始列，用于后续识别计算结果列
            initial_columns = set(chunk_df.columns)
            
            block_result = {
                "block_name": block_name,
                "compute_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_matched_cols": len(col_func_list),
                "success_cols": 0,
                "failed_cols": 0,
                "details": {"success": [], "failed": []}
            }
            child_cached_keys = set()
            original_index = chunk_df.index

            for col_name, func_list in col_func_list:
                computed_key = (block_name, col_name)
                # 检查缓存
                with cache_lock:
                    cache_hit = use_cache and not force_recompute and computed_key in computed_cache
                if cache_hit:
                    progress_counter.value += 1
                    continue

                try:
                    # 1. 基础校验
                    if col_name not in chunk_df.columns:
                        raise ValueError(f"原始列不存在")
                    
                    raw_col_data = chunk_df[col_name]
                    logger.debug(f"块[{block_name}]列[{col_name}]：原始数据行数={len(raw_col_data)}，非空行数={raw_col_data.notna().sum()}")

                    # 2. 解析列表列（本地核心逻辑）
                    is_list_column = raw_col_data.apply(lambda x: isinstance(x, list)).any()
                    if is_list_column:
                        logger.info(f"块[{block_name}]列[{col_name}]：检测到列表列，开始解析")
                        col_data = ChunkManager._parse_list_column(raw_col_data)
                        logger.info(f"块[{block_name}]列[{col_name}]：列表列解析完成，非空行数={col_data.notna().sum()}")
                    else:
                        # 3. 数据预处理（本地核心逻辑）
                        col_data = ChunkManager._preprocess_column_data(raw_col_data)

                    # 4. 缺失值处理（本地核心逻辑）
                    valid_data = ChunkManager._handle_missing_values(col_data, fillna_cfg)
                    if valid_data.notna().sum() == 0:
                        raise ValueError("缺失值处理后无有效数据")

                    # 5. 逐函数执行（本地核心逻辑）
                    for func in func_list:
                        func_name = func.__name__ if hasattr(func, "__name__") else "lambda_func"
                        new_col_name = f"{col_name}_{func_name}"
                        func_kwargs = {}  # 可根据实际需求扩展配置

                        # 函数兼容性校验
                        ChunkManager._check_function_compatibility(func, func_name, col_name)

                        # 安全执行元素级函数（本地核心逻辑）
                        result_valid = valid_data.apply(lambda x: ChunkManager._safe_execute_element(x, func, func_kwargs))

                        # 结果补全（本地核心逻辑）
                        result_padded = ChunkManager._pad_result(result_valid, original_index, padding_cfg)

                        # 写入新列
                        chunk_df[new_col_name] = result_padded
                        logger.info(f"块[{block_name}]列[{col_name}]：函数[{func_name}]执行成功，新增列={new_col_name}")

                    # 6. 更新缓存和结果
                    child_cached_keys.add(computed_key)
                    with cache_lock:
                        computed_cache[computed_key] = True
                    block_result["success_cols"] += 1
                    block_result["details"]["success"].append(col_name)

                except Exception as e:
                    error_msg = str(e)[:100]
                    block_result["failed_cols"] += 1
                    block_result["details"]["failed"].append({
                        "col_name": col_name,
                        "error": error_msg
                    })
                    logger.error(f"块[{block_name}]列[{col_name}]计算失败：{error_msg}")
                    
                    # 按错误处理策略执行
                    if error_handling == "abort":
                        raise RuntimeError(f"列[{col_name}]计算失败，中断流程") from e
                    elif error_handling == "retry":
                        try:
                            logger.info(f"块[{block_name}]列[{col_name}]：重试计算")
                            # 重试逻辑（与上面一致）
                            if is_list_column:
                                col_data = ChunkManager._parse_list_column(raw_col_data)
                            else:
                                col_data = ChunkManager._preprocess_column_data(raw_col_data)
                            valid_data = ChunkManager._handle_missing_values(col_data, fillna_cfg)
                            for func in func_list:
                                func_name = func.__name__ if hasattr(func, "__name__") else "lambda_func"
                                new_col_name = f"{col_name}_{func_name}"
                                result_valid = valid_data.apply(lambda x: ChunkManager._safe_execute_element(x, func, {}))
                                result_padded = ChunkManager._pad_result(result_valid, original_index, padding_cfg)
                                chunk_df[new_col_name] = result_padded
                            child_cached_keys.add(computed_key)
                            with cache_lock:
                                computed_cache[computed_key] = True
                            block_result["success_cols"] += 1
                            block_result["details"]["success"].append(col_name)
                            logger.info(f"块[{block_name}]列[{col_name}]：重试成功")
                        except Exception as retry_e:
                            logger.error(f"块[{block_name}]列[{col_name}]：重试失败：{str(retry_e)[:100]}")

                # 更新进度
                progress_counter.value += 1
            
            # 创建新DataFrame，只保留时间戳列和计算结果列
            # 1. 确定计算结果列（新添加的列）
            result_columns = list(set(chunk_df.columns) - initial_columns)
            
            # 2. 处理时间戳列
            if isinstance(base_time_columns, str):
                base_time_columns = [base_time_columns]
            available_time_columns = [col for col in base_time_columns if col in chunk_df.columns]
            
            # 3. 创建结果DataFrame
            result_df = pd.DataFrame(index=chunk_df.index)

            if keep_original_data:
                # 保留所有原始列 + 时间列 + 计算结果列
                all_columns = list(initial_columns) + available_time_columns + result_columns
                all_columns = list(dict.fromkeys(all_columns))  # 去重并保留顺序
            else:
                # 原有逻辑：仅保留时间列 + 计算结果列
                all_columns = available_time_columns + result_columns

            for col in all_columns:
                if col in chunk_df.columns:
                    result_df[col] = chunk_df[col]
            
            # 4. 替换原chunk_df
            chunk_df = result_df

            return block_name, block_result, chunk_df, child_cached_keys
        except Exception as e:
            logger.error(f"子进程：块[{block_name}]计算失败：{str(e)[:200]}", exc_info=True)
            raise

    def _compute_concurrently(
        self,
        block_funcs_mapping: Dict[str, Dict[str, List[Callable]]],
        n_workers: int = None,
        force_recompute: bool = False,
        error_handling: str = "skip",
        fillna_strategy: Dict[str, Any] = {'method': 'ignore', 'value': np.nan},
        padding_strategy: Dict[str, Any] = {'method': 'nan', 'value': np.nan},
        use_cache: bool = True,
        keep_original_data: bool = True, 
        global_progress_counter: Optional["multiprocessing.Manager.Value"] = None,
        show_progress: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
        """最终修复版：适配本地计算逻辑，解决序列化问题"""
        
        # Windows 平台适配
        import platform
        if platform.system() == "Windows":
            try:
                if multiprocessing.get_start_method(allow_none=True) is None:
                    multiprocessing.set_start_method("spawn", force=True)
                logger.info("Windows平台适配：已设置多进程启动方式为'spawn'")
            except RuntimeError as e:
                logger.warning(f"Windows平台多进程启动方式设置失败：{str(e)}，使用当前默认方式")

        # 输入校验
        if not isinstance(block_funcs_mapping, dict) or len(block_funcs_mapping) == 0:
            raise ValueError("block_funcs_mapping 必须是非空字典（块名→传感器-函数映射）")
        if not isinstance(use_cache, bool):
            raise TypeError("use_cache 必须是布尔值（True/False）")

        # 自动加载未加载的数据块（加锁保护）
        current_blocks = set(block_funcs_mapping.keys())
        with self.chunk_lock:
            unloaded_blocks = [block for block in current_blocks if block not in self.chunks]
        if unloaded_blocks:
            logger.info(f"检测到 {len(unloaded_blocks)} 个未加载的数据块：{unloaded_blocks}，开始自动加载...")
            failed_blocks = []
            for block_name in unloaded_blocks:
                try:
                    self.load_chunk(block_name)
                    logger.info(f"✅ 自动加载数据块 '{block_name}' 成功")
                except Exception as e:
                    logger.error(f"❌ 自动加载数据块 '{block_name}' 失败：{str(e)[:100]}", exc_info=True)
                    failed_blocks.append(block_name)
                    self.release_chunks(block_name)
            if failed_blocks:
                raise RuntimeError(f"以下数据块自动加载失败，无法继续计算：{failed_blocks}")
        else:
            logger.info("所有需要计算的数据块均已加载，无需自动加载")

        # 多进程参数优化
        max_cpu = multiprocessing.cpu_count()
        n_workers = n_workers or max_cpu
        n_workers = min(n_workers, max_cpu * 2)
        n_workers = max(n_workers, 1)
        n_blocks = len(block_funcs_mapping)
        logger.info(f"并发计算参数初始化完成：进程数={n_workers}，数据块数={n_blocks}，强制重算={force_recompute}，错误处理策略={error_handling}")

        # 缓存预处理（加锁保护）
        if not use_cache:
            with self.cache_lock:
                self._computed_cache.clear()
            logger.info(f"缓存已关闭（use_cache=False），已清空历史缓存，强制重算所有列")

        # 预过滤已计算的列（避免重复计算）
        pre_filtered_tasks = []
        block_col_funcs = {}
        total_cols_to_compute = 0
        for block_name, sensor_funcs in block_funcs_mapping.items():
            with self.chunk_lock:
                chunk_df = self.chunks.get(block_name)
                if chunk_df is None:
                    logger.warning(f"块[{block_name}]已被释放，跳过预过滤")
                    continue
                original_cols = chunk_df.columns.tolist()
            col_func_list = []
            for sensor_id, func_list in sensor_funcs.items():
                sensor_result = self._process_single_sensor(
                    sensor_id=sensor_id, func_list=func_list, original_cols=original_cols, is_main_process=True
                )
                if not sensor_result["valid"]:
                    logger.debug(f"块[{block_name}]传感器[{sensor_id}]无匹配列，跳过")
                    continue
                for real_col_name in sensor_result["matched_cols"]:
                    computed_key = (block_name, real_col_name)
                    with self.cache_lock:
                        if use_cache and not force_recompute and computed_key in self._computed_cache:
                            if global_progress_counter is not None:
                                global_progress_counter.value += 1
                            continue
                    col_func_list.append((real_col_name, func_list))
                    total_cols_to_compute += 1
            if col_func_list:
                pre_filtered_tasks.append(block_name)
                block_col_funcs[block_name] = col_func_list
                logger.info(f"块[{block_name}]预过滤完成，需计算列数：{len(col_func_list)}")
            else:
                logger.info(f"块[{block_name}]无需要计算的列（已缓存或无匹配列）")

        # 无计算任务直接返回
        if total_cols_to_compute == 0:
            logger.info(f"当前 {n_blocks} 个块均无需要计算的列（已缓存或无匹配列）")
            block_dfs = {}
            with self.chunk_lock:
                for block in block_funcs_mapping.keys():
                    if block in self.chunks:
                        block_dfs[block] = self.chunks[block].copy()
            return [], block_dfs

        # ------------------------------ 共享资源初始化 ------------------------------
        with self.manager as manager:
            # 进度计数器（多进程共享）
            if global_progress_counter is None:
                local_progress_counter = manager.Value('i', 0)
                global_progress_counter = local_progress_counter
            else:
                local_progress_counter = global_progress_counter

            # 共享缓存（多进程同步）
            computed_cache = manager.dict()
            with self.cache_lock:
                computed_cache.update(self._computed_cache)
            logger.info(f"已加载历史缓存：共 {len(computed_cache)} 个已计算列")

            # 文本进度显示（独立线程）
            progress_thread = None
            progress_stop_event = manager.Event()
            def print_text_progress():
                progress_length = 20
                while not progress_stop_event.is_set():
                    current = local_progress_counter.value
                    percent = min(100, (current / total_cols_to_compute) * 100) if total_cols_to_compute > 0 else 100
                    filled_length = int(progress_length * current // total_cols_to_compute)
                    progress_bar = "[" + "="*filled_length + " "*(progress_length - filled_length) + "]"
                    print(f"\r计算进度：{progress_bar} {percent:.1f}% 已完成{current}/{total_cols_to_compute}列", end="", flush=True)
                    time.sleep(0.5)
                print(f"\r计算进度：[{'='*progress_length}] 100% 已完成{total_cols_to_compute}/{total_cols_to_compute}列", flush=True)
                print("="*50 + " 计算完成 " + "="*50 + "\n")

            if show_progress:
                progress_thread = threading.Thread(target=print_text_progress, daemon=True)
                progress_thread.start()
                logger.info("启动文本式进度显示线程")

            # ------------------------------ 构造任务参数（显式传递所有依赖） ------------------------------
            task_args = []
            for block_name in pre_filtered_tasks:
                col_func_list = block_col_funcs[block_name]
                # 传递本地计算必需的配置（静态方法无法访问self）
                task_args.append((
                    block_name,
                    col_func_list,
                    force_recompute,
                    error_handling,
                    fillna_strategy,
                    padding_strategy,
                    self.chunks,      # 共享数据块
                    self.chunk_lock,  # 数据块锁
                    computed_cache,   # 共享缓存
                    self.cache_lock,  # 缓存锁
                    use_cache,
                    local_progress_counter,
                    self.column_separators,  # 列分隔符（本地逻辑依赖）
                    self.base_time_columns,   # 系统列（本地逻辑依赖）
                    keep_original_data       # 新增：传递保留原始数据参数
                ))

            # ------------------------------ 多进程执行（调用修复后的静态工作方法） ------------------------------
            try:
                logger.info(f"启动多进程计算：进程池大小={n_workers}，任务数={len(task_args)}")
                with multiprocessing.Pool(processes=n_workers) as pool:
                    results = pool.starmap_async(ChunkManager._compute_single_block_worker, task_args)
                    pool.close()
                    pool.join()
                    child_results = results.get()
                logger.info("所有多进程计算任务执行完成，开始收集结果")

                # 停止进度线程
                if show_progress and progress_thread.is_alive():
                    progress_stop_event.set()
                    progress_thread.join(timeout=2)
                    logger.info("进度显示线程已停止")

            except Exception as e:
                if show_progress and progress_thread.is_alive():
                    progress_stop_event.set()
                logger.error(f"并发计算失败：{str(e)[:200]}", exc_info=True)
                raise RuntimeError(f"并发计算失败：{str(e)[:200]}") from e

            # ------------------------------ 结果合并与缓存同步 ------------------------------
            all_block_results = []
            block_dfs = {}
            new_cached_keys = set()

            logger.info("开始合并计算结果...")
            for block_name, block_result, chunk_df, child_cached_keys in child_results:
                with self.chunk_lock:
                    self.chunks[block_name] = chunk_df
                block_dfs[block_name] = chunk_df.copy()
                all_block_results.append(block_result)
                new_cached_keys.update(child_cached_keys)
                logger.info(f"块[{block_name}]结果合并完成，成功列数：{block_result['success_cols']}，失败列数：{block_result['failed_cols']}")

            # 同步缓存
            with self.cache_lock:
                for key in new_cached_keys:
                    self._computed_cache[key] = True
            logger.info(f"缓存同步完成：新增{len(new_cached_keys)}个缓存项，当前缓存共 {len(self._computed_cache)} 个已计算列")

            # 处理无需计算的块
            for block_name in block_funcs_mapping.keys():
                if block_name not in block_dfs:
                    with self.chunk_lock:
                        if block_name in self.chunks:
                            block_dfs[block_name] = self.chunks[block_name].copy()
                    all_block_results.append({
                        "block_name": block_name,
                        "compute_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total_matched_cols": 0,
                        "success_cols": 0,
                        "failed_cols": 0,
                        "details": {"success": [], "failed": []}
                    })
                    logger.info(f"块[{block_name}]无计算任务，直接返回原始数据")

            # 打印汇总报告
            self._print_concurrent_summary(all_block_results)
            return all_block_results, block_dfs
    
    def release_chunks(self, names: Union[str, List[str], None] = None):
        """释放数据块（原逻辑不变，加锁保护）"""
        result = {
            'released_chunks': [],
            'skipped_names': []
        }

        with self.chunk_lock:
            if not self.chunks:
                logger.info("无已加载的数据块可释放")
                return result

            # 处理释放目标：None→释放所有，字符串→单个，列表→多个
            if names is None:
                target_names = list(self.chunks.keys())
            elif isinstance(names, str):
                target_names = [names]
            elif isinstance(names, list):
                target_names = names
            else:
                raise TypeError("names参数必须是str、List[str]或None")

            # 执行释放
            for name in target_names:
                if name in self.chunks:
                    del self.chunks[name]
                    result['released_chunks'].append(name)
                    logger.info(f"✅ 已释放数据块：{name}")
                else:
                    result['skipped_names'].append(name)
                    logger.warning(f"⚠️  数据块 '{name}' 未加载，跳过释放")

        # 触发垃圾回收
        gc.collect()
        logger.info(
            f"释放完成 → 成功释放: {len(result['released_chunks'])} 个，"
            f"跳过未加载: {len(result['skipped_names'])} 个"
        )
        return result

    def load_chunk(self, name: str, merge_with_core: bool = False):
        """加载数据块（原逻辑不变，加锁保护）"""
        with self.chunk_lock:
            if name in self.chunks:
                logger.info(f"数据块 '{name}' 已加载，直接返回副本")
                return self.chunks[name].copy()

        logger.info(f"开始加载数据块：{name}，merge_with_core={merge_with_core}")
        if merge_with_core:
            warnings.warn("merge_with_core参数已兼容（核心列逻辑已移除，无拼接效果）")
            logger.warning("merge_with_core参数已兼容（核心列逻辑已移除，无拼接效果）")

        # 查找文件路径（根据config的file_mappings配置）
        file_path = None
        if name in self.config.get('file_mappings', {}):
            rel_path = self.config['file_mappings'][name]
            file_path = os.path.join(self.base_dir, rel_path)
        else:
            # 传感器ID匹配对应的组数据块
            sensor_id = name
            sensor_groups = self.config.get('sensor_groups', {})
            group_name = None
            for group, sensors in sensor_groups.items():
                if isinstance(sensors, list) and any(str(sensor_id) == str(s) for s in sensors):
                    group_name = group
                    break
            if group_name and group_name in self.config['file_mappings']:
                rel_path = self.config['file_mappings'][group_name]
                file_path = os.path.join(self.base_dir, rel_path)
            else:
                raise ValueError(f"未找到传感器或组 '{name}' 的映射配置（检查config的file_mappings和sensor_groups）")

        # 验证文件存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载parquet文件（根据实际格式调整，如csv可改为pd.read_csv）
        try:
            chunk_data = pd.read_parquet(file_path)
            logger.info(f"成功读取数据块 '{name}'，形状: {chunk_data.shape}，列数: {len(chunk_data.columns)}")
        except Exception as e:
            raise ValueError(f"加载数据块 '{name}' 失败: {str(e)}") from e

        # 存入共享字典（加锁保护）
        with self.chunk_lock:
            self.chunks[name] = chunk_data
        logger.info(f"数据块 '{name}' 已存入共享缓存")
        return chunk_data.copy()

    # ------------------------------ 唯一对外核心接口 ------------------------------
    def compute(
        # 【解析函数映射参数】
        self,
        target_col: Union[str, List[str]] = None,
        funcs: Union[List[Callable], List[List[Callable]], List[Tuple[str, Callable]]] = None,
        func_mapping: Union[Dict[str, List[Union[Callable, Tuple[str, Callable]]]], List[Dict[str, Any]]] = None,
        keep_original_data: bool = False, 

        # 【并发计算参数】
        n_workers: int = None,
        force_recompute: bool = False,
        error_handling: str = "skip",
        fillna_strategy: Dict[str, Any] = {'method': 'ignore', 'value': np.nan},
        padding_strategy: Dict[str, Any] = {'method': 'nan', 'value': np.nan},
        use_cache: bool = True,
        global_progress_counter: Optional["multiprocessing.managers.ValueProxy"] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        统一对外核心接口：一键完成「函数映射解析→并发计算→结果拼接」全流程
        
        核心逻辑（严格遵循基础块拼接思想）：
            1. 以首个非空计算块为基础DataFrame：保留其所有有效时间列（self.base_time_columns中存在的列）+ 自身计算列
            2. 后续块仅提取计算列：彻底排除所有时间列（self.base_time_columns中所有列），不保留任何时间相关数据
            3. 直接按列拼接：基础块提供完整时间序列，后续块仅补充计算结果，无需额外对齐
            4. 最终结果：仅含首块的所有有效时间列 + 所有块的计算列，无任何后续块时间列残留
        
        参数分类：
            【1. 函数映射解析参数】（三选一模式：func_mapping 或 target_col+funcs）
                target_col: 简单模式 - 目标列标识（'all'/'core'、列列表、块列表(block:xxx)、通配符）
                funcs:       简单模式 - 函数列表（支持单个列表、列表的列表、带别名元组(alias, func)）
                func_mapping:批量/完整模式 - 字典（批量）或列表字典（完整）
            
            【2. 并发计算参数】
                n_workers:           进程数（默认：CPU核心数）
                force_recompute:     是否强制重算（默认：False，复用缓存）
                error_handling:      错误处理策略（默认：'skip'，跳过错误列）
                fillna_strategy:     缺失值填充策略（默认：{'method': 'ignore', 'value': np.nan}）
                padding_strategy:    结果补全策略（默认：{'method': 'nan', 'value': np.nan}）
                use_cache:           是否启用缓存（默认：True，关闭则强制重算）
                global_progress_counter: 全局进度计数器（用于逐块调度时的整体进度）
                show_progress:       是否显示进度条（默认：True，False 关闭进度条）
        
        返回值：
            pd.DataFrame: 拼接后的完整结果DataFrame，包含首块所有有效时间列和所有计算结果列。
        """
        logger.info("="*80)
        logger.info("开始统一计算流程：函数映射解析 → 并发计算 → 结果拼接")
        logger.info("="*80)

        try:
            # 步骤1：解析函数映射
            logger.info("第一步：解析函数映射...")
            block_funcs_mapping = self._parse_function_mapping(
                target_col=target_col,
                funcs=funcs,
                func_mapping=func_mapping
            )

            # 步骤2：并发计算
            logger.info("第二步：启动并发计算...")
            compute_results, block_dfs = self._compute_concurrently(
                block_funcs_mapping=block_funcs_mapping,
                n_workers=n_workers,
                force_recompute=force_recompute,
                error_handling=error_handling,
                fillna_strategy=fillna_strategy,
                padding_strategy=padding_strategy,
                use_cache=use_cache,
                global_progress_counter=global_progress_counter,
                show_progress=show_progress,
                keep_original_data = keep_original_data
            )
            
            # 步骤3：列拼接结果（首块保留全部有效时间列，后续块仅保留计算列）
            logger.info("第三步：以首块为基础，按列拼接计算结果（严格过滤后续块时间列）...")
            
            # -------------------------- 核心：定义时间列范围 --------------------------
            # 所有需要过滤的时间列（来自配置，后续块需彻底排除）
            all_time_cols = self.base_time_columns
            if not isinstance(all_time_cols, list):
                all_time_cols = [all_time_cols]
            all_time_cols = list(set(all_time_cols))  # 去重
            logger.info(f"需要过滤的所有时间列：{all_time_cols}")
            
            # -------------------------- 步骤1：筛选基础块（首块保留全部有效时间列） --------------------------
            base_df = None
            remaining_blocks = {}
            
            for block_name, df in block_dfs.items():
                if df.empty:
                    logger.warning(f"块 [{block_name}] 为空，跳过")
                    continue
                
                # 1. 提取首块的有效时间列（存在于当前块的all_time_cols）
                base_valid_time_cols = [col for col in all_time_cols if col in df.columns]
                if not base_valid_time_cols:
                    logger.warning(f"块 [{block_name}] 无有效时间列（all_time_cols中列均不存在），跳过")
                    continue
                
                # 2. 提取当前块的计算列（排除所有时间列）
                calc_cols = [col for col in df.columns if col not in all_time_cols]
                
                # 3. 构造当前块的“有效数据”（时间列 + 计算列）
                valid_cols = base_valid_time_cols + calc_cols
                block_valid_df = df[valid_cols].copy()
                
                if base_df is None:
                    # 首个有效块作为基础块：保留其全部有效时间列 + 自身计算列
                    base_df = block_valid_df
                    logger.info(f"使用首个非空块 [{block_name}] 作为基础，形状={base_df.shape}")
                    logger.info(f"基础块保留的时间列：{base_valid_time_cols}")
                    logger.info(f"基础块计算列数：{len(calc_cols)}")
                    logger.info(f"基础块完整列：{base_df.columns.tolist()}")
                else:
                    # 其他块暂存（仅含计算列，后续统一过滤时间列）
                    remaining_blocks[block_name] = df.copy()
            
            # 处理没有有效基础块的情况
            if base_df is None:
                logger.warning("没有找到含有效时间列的数据块，返回空DataFrame")
                return pd.DataFrame(columns=all_time_cols)
            
            # -------------------------- 步骤2：拼接后续块（仅保留计算列，彻底排除时间列） --------------------------
            result_df = base_df.copy()
            added_columns = set(result_df.columns)  # 已添加的列（首块时间列+计算列）
            duplicate_columns = []
            
            for block_name, df in remaining_blocks.items():
                logger.info(f"处理块 [{block_name}]，原始形状={df.shape}")
                
                # 核心过滤：仅提取计算列（排除所有时间列，不保留任何时间相关数据）
                block_calc_cols = [col for col in df.columns if col not in all_time_cols]
                if not block_calc_cols:
                    logger.warning(f"块 [{block_name}] 无有效计算列（所有列均为时间列），跳过")
                    continue
                
                block_calc_df = df[block_calc_cols].copy()
                
                # 处理列名冲突（避免覆盖已有列）
                rename_mapping = {}
                for col in block_calc_cols:
                    if col in added_columns:
                        duplicate_columns.append(col)
                        new_col_name = f"{col}_{block_name}"
                        rename_mapping[col] = new_col_name
                        logger.warning(f"列名冲突: '{col}' 已存在（来自基础块/其他块），重命名为 '{new_col_name}'")
                
                if rename_mapping:
                    block_calc_df.rename(columns=rename_mapping, inplace=True)
                
                # 按列拼接（仅补充计算列，不添加任何时间列）
                pre_shape = result_df.shape
                result_df = pd.concat([result_df, block_calc_df], axis=1, ignore_index=False)
                post_shape = result_df.shape
                logger.info(f"  块 [{block_name}] 拼接完成: 行数 {pre_shape[0]}→{post_shape[0]}, 列数 {pre_shape[1]}→{post_shape[1]}")
                
                # 更新已添加列集合
                added_columns.update(block_calc_df.columns)
            
            # -------------------------- 步骤3：最终校验（确保无后续块时间列残留） --------------------------
            # 检查结果中是否有后续块的时间列（理论上不会有，兜底清理）
            unexpected_time_cols = [col for col in result_df.columns if col in all_time_cols and col not in base_df.columns]
            if unexpected_time_cols:
                result_df = result_df.drop(columns=unexpected_time_cols)
                logger.warning(f"兜底清理：移除后续块残留时间列：{unexpected_time_cols}")
            
            # 重置索引（保持整洁，不改变数据顺序）
            result_df = result_df.reset_index(drop=True)
            
            # -------------------------- 输出结果摘要 --------------------------
            final_time_cols = [col for col in result_df.columns if col in all_time_cols]
            final_calc_cols = [col for col in result_df.columns if col not in all_time_cols]
            
            logger.info("="*60)
            logger.info("最终结果摘要：")
            logger.info(f"总数据量：{result_df.shape[0]} 行 × {result_df.shape[1]} 列")
            logger.info(f"保留的时间列（全部来自首块）：{final_time_cols}（共{len(final_time_cols)}列）")
            logger.info(f"计算列总数：{len(final_calc_cols)} 列")
            logger.info(f"列名冲突总数：{len(duplicate_columns)} 个（已重命名）")
            logger.info("="*60)
            
            # 计算结果统计（可选）
            total_success = sum(res["success_cols"] for res in compute_results)
            total_failed = sum(res["failed_cols"] for res in compute_results)
            logger.info(f"计算统计：成功 {total_success} 列, 失败 {total_failed} 列")
            if total_failed > 0:
                failed_details = []
                for res in compute_results:
                    for fail in res["details"]["failed"]:
                        failed_details.append(f"{res['block_name']}/{fail['col_name']}: {fail['error'][:50]}")
                logger.warning(f"失败详情（前5条）:\n" + "\n".join(failed_details[:5]))
            
            logger.info("="*80)
            logger.info("统一计算流程完成！")
            logger.info("="*80)
            return result_df

        except Exception as e:
            logger.error("统一计算流程失败：", exc_info=True)
            raise RuntimeError(f"统一计算流程失败：{str(e)[:200]}") from e

