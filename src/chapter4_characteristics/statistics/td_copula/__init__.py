"""
窗间时序 Copula（与静态 copula 流水线并行，不覆盖其结果）。

分层：
  1. 窗内静态联合（已有 copula）— 同窗特征相依
  2. 窗间 Markov（本包）— 相邻窗特征轨迹 / 长时 MC
  3. 窗内过程（CLI stage=intra 口子）— 与雨流对齐；本阶段未实现
"""

from src.chapter4_characteristics.statistics.td_copula.pipeline import run_td_copula_job

__all__ = ["run_td_copula_job"]
