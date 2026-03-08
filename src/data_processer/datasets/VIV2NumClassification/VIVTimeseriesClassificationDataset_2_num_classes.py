import torch
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, Union, Literal, List
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_processer.datasets.VIV2NumClassification.BaseDataset import BaseDataset
from config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import VIVTimeSeriesClassificationDatasetConfig 


class VIVTimeSeriesClassificationDataset(BaseDataset):
    """
    VIV时序分类数据集（严格兼容基类抽象方法，优雅适配LSTM/CNN双模式）
    核心特性：
    1. 抽象方法合规：严格实现基类`_parse_sample`和`__getitem__`抽象方法；
    2. 归一化修复：基于训练集全局统计量，避免单样本归一化混乱；
    3. 双模式解耦：清晰区分时序/网格模式，接口符合PyTorch网络规范；
    4. 基类逻辑复用：复用缓存、路径管理、数据集划分等基类通用能力；
    5. 代码低冗余：统一样本处理流水线，消除训练/验证/测试集重复逻辑；
    6. 二分类适配：剔除标签1（过渡）样本，标签2（异常）转为1，保留标签0（一般）不变。
    """
    def __init__(self, config: VIVTimeSeriesClassificationDatasetConfig):
        # 1. 先调用基类初始化（必须优先执行，完成路径/划分/缓存初始化）
        super().__init__(config)
        
        # 2. 解析核心配置，区分通用配置和双模式配置（兼容基类配置）
        self._parse_base_config(config)
        
        # --------------------------
        # 新增：初始化时处理二分类（不改动原有结构，嵌入在此处）
        # 时机：logger已初始化，路径已加载，适合执行样本过滤
        # --------------------------
        self._filter_transition_samples()
        
        # 3. 解析模式配置（原有步骤，保持不变）
        self._parse_mode_config(config)
        
        # 4. 计算训练集全局归一化统计量（修复原单样本归一化漏洞，基于过滤后样本）
        self.global_norm_stats = self._calc_global_normalize_stats()

    def _parse_base_config(self, config: VIVTimeSeriesClassificationDatasetConfig):
        """解析通用基础配置，与输出模式无关，对齐基类配置（修复日志初始化问题）"""
        # 时序处理配置
        self.batch_first = config.batch_first
        self.fix_seq_len = config.fix_seq_len
        self.pad_mode = config.pad_mode
        self.trunc_mode = config.trunc_mode
        self.normalize = config.normalize
        self.normalize_type = config.normalize_type
        self.ts_mean = config.ts_mean
        self.ts_std = config.ts_std
        self.feat_dim = config.feat_dim

        # 复用基类缓存配置，不自定义缓存键
        self.logger = logging.getLogger(__name__)
        
        # --------------------------
        # 修复日志初始化错误：兜底配置日志处理器，避免无输出/报错
        # --------------------------
        if not self.logger.handlers:
            # 添加控制台日志处理器
            console_handler = logging.StreamHandler()
            # 定义日志格式
            log_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)
            # 默认日志级别设为INFO（可根据需要调整）
            self.logger.setLevel(logging.INFO)

    def _filter_transition_samples(self):
        """
        初始化时过滤过渡样本（标签1），不改动原有结构，仅更新路径列表
        功能：剔除所有标签为1的样本，更新train/val/test/full路径集合
        """
        self.logger.info("开始过滤标签为1的过渡状态样本，构建二分类数据集...")

        # 内部过滤函数，避免重复代码
        def _filter_single_path_list(path_list: List[Path]) -> List[Path]:
            valid_paths = []
            transition_count = 0
            for file_path in path_list:
                try:
                    # 仅解析原始标签用于过滤，不做其他预处理
                    _, raw_label = self._parse_raw_sample(file_path)
                    if raw_label != 1:  # 仅保留标签0和2的样本
                        valid_paths.append(file_path)
                    else:
                        transition_count += 1
                        self.logger.debug(f"剔除过渡样本：{file_path.name}（标签=1）")
                except Exception as e:
                    self.logger.warning(f"解析文件{file_path.name}失败，自动跳过：{str(e)}")
            self.logger.info(
                f"路径过滤统计：原始{len(path_list)}个 → 有效{len(valid_paths)}个，剔除过渡样本{transition_count}个"
            )
            return valid_paths

        # 更新所有数据集路径（保持原有路径属性名称，不改动后续逻辑）
        self.train_paths = _filter_single_path_list(self.train_paths)
        self.val_paths = _filter_single_path_list(self.val_paths)
        self.test_paths = _filter_single_path_list(self.test_paths)
        self.full_file_paths = _filter_single_path_list(self.full_file_paths)

        # 空数据集校验警告
        for mode, paths in [
            ("训练集", self.train_paths),
            ("验证集", self.val_paths),
            ("测试集", self.test_paths),
            ("完整数据集", self.full_file_paths)
        ]:
            if len(paths) == 0:
                self.logger.warning(f"⚠️ {mode}过滤后为空，请检查原始数据是否包含标签0/2的有效样本")

    def _parse_mode_config(self, config: VIVTimeSeriesClassificationDatasetConfig):
        """解析双输出模式配置，严格区分时序/视觉网络接口（原有代码，无改动）"""
        # 输出模式校验（确保仅支持两种合法模式）
        self.output_mode = config.output_mode
        if self.output_mode not in ["time_series", "grid_2d"]:
            raise ValueError(f"无效输出模式：{self.output_mode}，仅支持'time_series'/'grid_2d'")
        
        # 网格模式固定配置（视觉CNN适配：通道优先格式）
        self.grid_h, self.grid_w = 50, 60
        self.grid_flat_len = self.grid_h * self.grid_w
        self.grid_channels = 1  # 单通道灰度图，符合PyTorch视觉接口规范
        
        # 时序模式目标长度（网格模式强制为3000，时序模式使用配置值）
        self.target_seq_len = (
            self.grid_flat_len if self.output_mode == "grid_2d" 
            else self.fix_seq_len
        )

    def _calc_global_normalize_stats(self) -> Union[dict, None]:
        """
        计算训练集全局归一化统计量（修复原单样本归一化混乱问题）
        仅在初始化时执行一次，所有样本（train/val/test）共用该统计量
        """
        if not self.normalize:
            return None
        
        self.logger.info("正在计算训练集全局归一化统计量...")
        all_train_ts = []
        
        # 遍历训练集所有样本，收集原始时序数据（不做任何预处理）
        for file_path in self.train_paths:
            raw_ts, _ = self._parse_raw_sample(file_path)
            all_train_ts.append(raw_ts)
        
        # 合并所有训练集时序数据，按特征维度计算统计量
        all_train_ts = np.vstack(all_train_ts)
        norm_stats = {}

        if self.normalize_type == "min-max":
            norm_stats["min"] = all_train_ts.min(axis=0, keepdims=True)
            norm_stats["max"] = all_train_ts.max(axis=0, keepdims=True)
            self.logger.info(f"全局Min-Max统计量：min={norm_stats['min'].squeeze()}, max={norm_stats['max'].squeeze()}")
        
        elif self.normalize_type == "z-score":
            if self.ts_mean is not None and self.ts_std is not None:
                norm_stats["mean"] = np.array(self.ts_mean).reshape(1, -1)
                norm_stats["std"] = np.array(self.ts_std).reshape(1, -1)
            else:
                norm_stats["mean"] = all_train_ts.mean(axis=0, keepdims=True)
                norm_stats["std"] = all_train_ts.std(axis=0, keepdims=True)
            self.logger.info(f"全局Z-Score统计量：mean={norm_stats['mean'].squeeze()}, std={norm_stats['std'].squeeze()}")
        
        return norm_stats

    def _parse_raw_sample(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        纯原始样本解析：仅加载mat文件，不做归一化/长度调整，保证统计量纯净
        为`_parse_sample`提供底层支撑，避免与基类抽象方法冲突
        """
        try:
            mat_data = loadmat(file_path)
            # 筛选有效时序数据（排除系统键，优先匹配显式键）
            valid_keys = [k for k in mat_data.keys() if not k.startswith("__") and isinstance(mat_data[k], np.ndarray)]
            if not valid_keys:
                raise ValueError("无有效非系统键的ndarray数据")
            
            # 优先提取标签和时序数据（增强鲁棒性，兼容显式键+原逻辑）
            ts_data, label = None, None
            if "label" in mat_data and "ts_data" in mat_data:
                # 显式键匹配（推荐格式，标签独立存储）
                ts_data = mat_data["ts_data"]
                label = int(mat_data["label"].squeeze())
            else:
                # 兼容原逻辑：第一个有效ndarray作为时序，键名作为标签
                first_key = valid_keys[0]
                ts_data = mat_data[first_key]
                label = int(first_key)

            # 数据预处理的时候，引入的差异
            ts_data = ts_data.squeeze()
            # 维度规整：确保为2D (seq_len, feat_dim)
            if ts_data.ndim == 1:
                ts_data = ts_data.reshape(-1, 1)
            elif ts_data.ndim > 2:
                ts_data = ts_data.squeeze()
            if ts_data.ndim != 2:
                raise ValueError(f"时序数据维度异常，需为2D，当前为{ts_data.ndim}D")

            # 特征维度校验
            actual_feat_dim = ts_data.shape[1]
            if self.feat_dim is not None and actual_feat_dim != self.feat_dim:
                raise ValueError(
                    f"特征维度不匹配：配置={self.feat_dim}，实际={actual_feat_dim}"
                )

            return ts_data.astype(np.float32), label

        except Exception as e:
            raise RuntimeError(f"解析文件{file_path.name}失败：{str(e)}")

    # --------------------------
    # 必须实现：基类抽象方法 _parse_sample
    # --------------------------
    def _parse_sample(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        实现基类抽象方法：解析单个样本并完成预处理（归一化+长度调整）
        新增：二分类标签转换（标签2→1，标签0保持不变，已过滤标签1）
        Args:
            file_path: 样本文件路径（Path对象）
        Returns:
            processed_ts: 预处理后的时序数据 (target_seq_len, feat_dim)
            label: 二分类样本标签（0=一般，1=异常）
        """
        # 步骤1：解析原始样本
        raw_ts, label = self._parse_raw_sample(file_path)
        
        # --------------------------
        # 新增：二分类标签转换（不改动原有步骤顺序，仅插入此处）
        # --------------------------
        label = 1 if label == 2 else 0  # 标签2转为1，标签0保持不变
        
        # 步骤2：全局归一化（使用训练集统计量，保证一致性）
        if self.normalize:
            raw_ts = self._normalize_sample(raw_ts)
        
        # 步骤3：固定序列长度（补全/截断，网格模式强制3000）
        if self.target_seq_len is not None:
            raw_ts = self._adjust_seq_length(raw_ts)
        
        return raw_ts, label

    def _normalize_sample(self, ts_data: np.ndarray) -> np.ndarray:
        """使用训练集全局统计量归一化单个样本，修复原单样本归一化漏洞"""
        if self.normalize_type == "min-max":
            min_val = self.global_norm_stats["min"]
            max_val = self.global_norm_stats["max"]
            ts_data = (ts_data - min_val) / (max_val - min_val + 1e-8)
        
        elif self.normalize_type == "z-score":
            mean_val = self.global_norm_stats["mean"]
            std_val = self.global_norm_stats["std"]
            ts_data = (ts_data - mean_val) / (std_val + 1e-8)
        
        return ts_data

    def _adjust_seq_length(self, ts_data: np.ndarray) -> np.ndarray:
        """统一调整序列长度，支持补全和截断，兼容双模式"""
        seq_len, feat_dim = ts_data.shape
        target_len = self.target_seq_len

        # 截断：长序列缩短至目标长度
        if seq_len > target_len:
            if self.trunc_mode == "head":
                ts_data = ts_data[:target_len, :]
            else:  # tail（默认，保留最新时序）
                ts_data = ts_data[-target_len:, :]
        
        # 补全：短序列延长至目标长度
        elif seq_len < target_len:
            pad_len = target_len - seq_len
            pad_data = self._generate_pad_data(pad_len, feat_dim, ts_data)
            ts_data = np.vstack([ts_data, pad_data])
        
        return ts_data

    def _generate_pad_data(self, pad_len: int, feat_dim: int, ts_data: np.ndarray) -> np.ndarray:
        """根据配置生成补全数据，避免重复逻辑"""
        if self.pad_mode == "zero":
            return np.zeros((pad_len, feat_dim), dtype=np.float32)
        elif self.pad_mode == "repeat":
            return np.repeat(ts_data[-1:], pad_len, axis=0)
        elif self.pad_mode == "mean":
            feat_mean = ts_data.mean(axis=0, keepdims=True)
            return np.repeat(feat_mean, pad_len, axis=0)
        else:
            raise ValueError(f"无效补全模式：{self.pad_mode}，仅支持'zero'/'repeat'/'mean'")

    def _process_time_series(self, ts_data: np.ndarray) -> torch.Tensor:
        """
        时序模式处理：适配LSTM时序网络输入格式
        Returns:
            ts_tensor: (seq_len, feat_dim) 或 (feat_dim, seq_len)（batch_first=False）
        """
        ts_tensor = torch.from_numpy(ts_data).float()
        # 调整维度顺序以适配batch_first配置
        if not self.batch_first:
            ts_tensor = ts_tensor.permute(1, 0)  # (seq_len, feat_dim) → (feat_dim, seq_len)
        return ts_tensor

    def _process_grid_2d(self, ts_data: np.ndarray) -> torch.Tensor:
        """
        网格模式处理：适配CNN视觉网络输入格式（通道优先，符合PyTorch规范）
        Returns:
            grid_tensor: (C, H, W) 单通道灰度图格式
        """
        seq_len, feat_dim = ts_data.shape
        
        # 严格校验（网格模式前置条件）
        if seq_len != self.grid_flat_len:
            raise ValueError(f"网格模式要求时序长度为{self.grid_flat_len}，当前为{seq_len}")
        if feat_dim > 1:
            raise NotImplementedError("网格模式暂仅支持单特征时序（feat_dim=1），多特征需扩展为多通道")
        
        # 重塑为网格并添加通道维度（视觉网络标准输入格式）
        grid_data = ts_data.reshape(self.grid_h, self.grid_w)
        grid_tensor = torch.from_numpy(grid_data).float().unsqueeze(0)  # (H, W) → (C, H, W)
        
        return grid_tensor

    def _get_sample(self, index: int, paths: List[Path]) -> Tuple[torch.Tensor, int]:
        """
        通用样本获取逻辑，复用基类缓存接口，支持双模式转换
        Args:
            index: 样本索引
            paths: 对应数据集（train/val/test）的文件路径列表
        Returns:
            模型输入张量（时序/网格格式）、样本标签
        """
        # 步骤1：优先读取基类内存缓存
        cached_sample = self._get_cached_sample(index)
        if cached_sample is not None:
            return cached_sample

        # 步骤2：获取文件路径并解析预处理样本（调用实现的抽象方法）
        file_path = paths[index]
        processed_ts, label = self._parse_sample(file_path)

        # 步骤3：按输出模式转换为模型输入张量
        if self.output_mode == "time_series":
            model_input = self._process_time_series(processed_ts)
        else:  # grid_2d
            model_input = self._process_grid_2d(processed_ts)

        # 步骤4：写入基类内存缓存（复用基类方法，无需自定义缓存键）
        self._cache_sample(index, (model_input, label))

        return model_input, label

    # --------------------------
    # 必须实现：基类抽象方法 __getitem__
    # --------------------------
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        实现基类抽象方法：获取当前实例模式对应的样本（时序/网格格式）
        根据基类`_dataset_mode`自动匹配train/val/test路径，无需手动指定
        """
        mode_path_map = {
            "full": self.full_file_paths,
            "train": self.train_paths,
            "val": self.val_paths,
            "test": self.test_paths
        }
        current_paths = mode_path_map[self._dataset_mode]
        return self._get_sample(index, current_paths)

    # --------------------------
    # 兼容原有接口：验证集/测试集样本获取
    # --------------------------
    def get_val_item(self, index: int) -> Tuple[torch.Tensor, int]:
        """获取验证集样本（复用通用逻辑，兼容原有接口）"""
        if index >= len(self.val_paths):
            raise IndexError(f"验证集索引{index}超出范围，验证集长度{len(self.val_paths)}")
        return self._get_sample(index, self.val_paths)

    def get_test_item(self, index: int) -> Tuple[torch.Tensor, int]:
        """获取测试集样本（复用通用逻辑，兼容原有接口）"""
        if index >= len(self.test_paths):
            raise IndexError(f"测试集索引{index}超出范围，测试集长度{len(self.test_paths)}")
        return self._get_sample(index, self.test_paths)

    # --------------------------
    # 扩展便捷接口：便于外部获取数据集信息
    # --------------------------
    def get_input_shape(self) -> Tuple[int, ...]:
        """获取模型输入形状，便于模型初始化，兼容双模式"""
        if self.output_mode == "time_series":
            if self.batch_first:
                return (self.target_seq_len, self.feat_dim)
            else:
                return (self.feat_dim, self.target_seq_len)
        else:  # grid_2d
            return (self.grid_channels, self.grid_h, self.grid_w)