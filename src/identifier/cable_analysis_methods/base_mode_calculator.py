from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# YAML 配置文件路径（相对于项目根目录）
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CABLE_PARAMS_YAML = _PROJECT_ROOT / "config" / "cables" / "cable_params.yaml"


@lru_cache(maxsize=1)
def _load_cable_config() -> dict:
    if not _CABLE_PARAMS_YAML.exists():
        warnings.warn(
            f"拉索参数配置文件不存在：{_CABLE_PARAMS_YAML}，所有查询将回退到默认参数。",
            stacklevel=3,
        )
        return {}
    with open(_CABLE_PARAMS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _params_from_cable(cable_id: str) -> Optional[dict]:
    cfg = _load_cable_config()
    cables = cfg.get("cables", {})
    return cables.get(cable_id)


def _cable_id_from_sensor(sensor_id: str) -> Optional[str]:
    cfg = _load_cable_config()
    return cfg.get("sensor_cable_map", {}).get(sensor_id)


# 默认参数（与原始硬编码保持完全一致，用于兼容旧调用）
_DEFAULT_PARAMS = dict(
    length=576.9147,
    force=8046000,
    alphac=20.338908862 * np.pi / 180,
    A=0.024092,
    m=100.8,
    E=2.1e11,
)


class Cal_Mount():
    """
    拉索理论模态频率计算器。

    旧调用方式（默认参数，完全兼容）：
        mount = Cal_Mount()

    按拉索编号查询（需 config/cables/cable_params.yaml 中有对应条目）：
        mount = Cal_Mount.from_cable("C34")

    按传感器 ID 查询（自动解析拉索编号）：
        mount = Cal_Mount.from_sensor("ST-VIC-C34-101-01")

    若配置文件中找不到对应条目，将打印 Warning 并回退到默认参数。
    """

    def __init__(self,
                 length=576.9147,
                 force=8046000,
                 alphac=20.338908862 * np.pi / 180,
                 A=0.024092,
                 m=100.8,
                 E=2.1e11):

        # 几何参数
        self.alphac = alphac    # 弧度制
        self.A = A
        self.length = length

        # 力学参数
        self.E = E
        self.force = force
        self.m = m
        self.multi_nums = self.base_modes()[1][-1] - self.base_modes()[1][-2]


        return

    # ------------------------------------------------------------------ #
    #  工厂方法：按拉索编号 / 传感器 ID 构造实例                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cable(cls, cable_id: str) -> "Cal_Mount":
        raw = _params_from_cable(cable_id)
        if raw is None:
            warnings.warn(
                f"[Cal_Mount] 配置文件中未找到拉索 '{cable_id}'，回退到默认参数。",
                stacklevel=2,
            )
            return cls(**_DEFAULT_PARAMS)

        missing = [k for k in ("length", "force", "alphac_deg", "E", "A", "m")
                   if raw.get(k) in (None, "???")]
        if missing:
            warnings.warn(
                f"[Cal_Mount] 拉索 '{cable_id}' 以下参数尚未填写：{missing}，回退到默认参数。",
                stacklevel=2,
            )
            return cls(**_DEFAULT_PARAMS)

        return cls(
            length=float(raw["length"]),
            force=float(raw["force"]),
            alphac=float(raw["alphac_deg"]) * np.pi / 180.0,
            A=float(raw["A"]),
            m=float(raw["m"]),
            E=float(raw["E"]),
        )

    @classmethod
    def from_sensor(cls, sensor_id: str) -> "Cal_Mount":
        cable_id = _cable_id_from_sensor(sensor_id)
        if cable_id is None:
            warnings.warn(
                f"[Cal_Mount] 传感器 '{sensor_id}' 未在 sensor_cable_map 中登记，回退到默认参数。",
                stacklevel=2,
            )
            return cls(**_DEFAULT_PARAMS)
        return cls.from_cable(cable_id)

    # ------------------------------------------------------------------ #

    def inplane_mode(self, n):

        # 奇数
        # 奇数对应正对称振动
        if n % 2 == 1:
            lambda_ = self.E * self.A * np.square((self.m * 9.8 * np.cos(self.alphac))) / (self.force ** 3)
            if n == 1:
                zeta = 0.0017 * np.power(lambda_, 2) + 0.1254 * lambda_ + 3.1444
            elif n == 3:
                zeta = 0.0053 * lambda_ + 9.4239
            else:
                zeta = n * np.pi
                
            fn = (zeta / (2 * np.pi * self.length)) * np.sqrt(self.force / self.m)

            return fn
        

        # 偶数
        # 偶数对应反对称振动
        elif n % 2 == 0:
            fn = (n / (2 * self.length)) * np.sqrt(self.force / self.m)

            return fn
        
    def outplane_mode(self, n):

        fn = (n / (2 * self.length)) * np.sqrt(self.force / self.m)

        return fn

    def base_modes(self, max_freq = 25):
        """
        调用该方法，传回小于max_freq的基频

        return inplane_modes, outplane_modes
        """
        
        inplane_modes = [0]
        outplane_modes = [0]

        # 面内
        while inplane_modes[-1] < max_freq:
            inplane_modes.append(self.inplane_mode(len(inplane_modes) + 1))


        # 面外
        while outplane_modes[-1] < max_freq:
            outplane_modes.append(self.outplane_mode(len(outplane_modes) + 1))
        
        self.inplane_modes = inplane_modes[1: ]
        self.outplane_modes = outplane_modes[1: ]

        return self.inplane_modes, self.outplane_modes
    
    def peaks(self, fx, pxxden, return_intervals = False):
        """
        返回能找到的前50阶振动模态
        return fx_peaks, pxxden_peaks

        if return_intervals = True:
            return fx_peaks, pxxden_peaks, fx_intervals, pxxden_intervals
        """
        
        # 临近模态区间范围
        interval_length = 0.2

        fx_peaks = []
        pxxden_peaks = []
        
        # 稍微兼容一下
        if len(fx.shape) == 2:
            fx = fx[0, :]
        if len(pxxden.shape) == 2:
            pxxden = pxxden[0, :]

        fxi = fx
        pxxdeni = pxxden
        
        if return_intervals:
            fx_intervals = []   
            pxxden_intervals = []

        pxxden_max_max = np.max(pxxden)
        max_value = pxxden_max_max
        while len(fxi) != 0:
            
            # 最大值
            max_value = np.max(pxxdeni)
            max_index = np.argwhere(pxxdeni == max_value)
            fx_max = fxi[max_index].item()

            # 获取该数值区间下的最大值
            fx_peaks.append(fx_max)
            pxxden_peaks.append(max_value)

            # 若需要返回区间
            if return_intervals:
                
                pxxden_intervali = pxxdeni[fxi < fx_max + self.multi_nums * interval_length]
                fx_intervali = fxi[fxi < fx_max + self.multi_nums * interval_length]

                pxxden_intervali = pxxden_intervali[fx_intervali > fx_max - self.multi_nums * interval_length]
                fx_intervali = fx_intervali[fx_intervali > fx_max - self.multi_nums * interval_length]

                fx_intervals.append(fx_intervali)
                pxxden_intervals.append(pxxden_intervali)


            # 删除区间
            # 左区间
            pxxden_left = pxxdeni[fxi < fx_max - self.multi_nums * interval_length]
            fx_left = fxi[fxi < fx_max - self.multi_nums * interval_length]

            # 右区间
            pxxden_right = pxxdeni[fxi > fx_max + self.multi_nums * interval_length]
            fx_right = fxi[fxi > fx_max + self.multi_nums * interval_length]

            pxxdeni = np.concatenate((pxxden_left, pxxden_right))
            fxi = np.concatenate((fx_left, fx_right))

        if return_intervals:
            return fx_peaks, pxxden_peaks, fx_intervals, pxxden_intervals
        else:
                return fx_peaks, pxxden_peaks
