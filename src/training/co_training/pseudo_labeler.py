from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_VIV_CLASS    = 1
_NORMAL_CLASS = 0


@dataclass
class PseudoLabel:
    sample_idx: int
    class_id:   int
    source:     str
    confidence: float


class PseudoLabeler:
    """
    协同置信度筛选器：结合 MECC 连续分值和 DL softmax 概率生成伪标签。

    置信度策略
    ----------
    MECC 高置信 VIV    : mecc_score < mecc_conf_viv
    MECC 高置信 Normal : mecc_score > mecc_conf_nor
    DL   高置信 VIV    : P(VIV) > dl_conf_viv
    DL   高置信 Normal : P(Normal) > dl_conf_nor

    合并规则（优先级依次降低）
    -------------------------
    1. MECC 高置信 VIV  且 DL 高置信 VIV  → source="both",  confidence=均值
    2. MECC 高置信 Nor  且 DL 高置信 Nor  → source="both",  confidence=均值
    3. 仅 DL 高置信 VIV（MECC 未检出）    → source="dl",   confidence=dl_viv_p
       （MECC 漏召回恢复场景）
    4. 仅 MECC 高置信 VIV（DL 不确定）    → source="mecc", confidence=1-mecc
    5. 仅 MECC 高置信 Nor（DL 不确定）    → source="mecc", confidence=mecc-nor_thr
    """

    def __init__(
        self,
        mecc_conf_viv: float = 0.15,
        mecc_conf_nor: float = 0.50,
        dl_conf_viv:   float = 0.92,
        dl_conf_nor:   float = 0.92,
    ):
        self.mecc_conf_viv = mecc_conf_viv
        self.mecc_conf_nor = mecc_conf_nor
        self.dl_conf_viv   = dl_conf_viv
        self.dl_conf_nor   = dl_conf_nor

    def generate(
        self,
        mecc_scores:  Dict[int, Tuple[int, float, Optional[float]]],
        dl_probas:    Dict[int, np.ndarray],
        gold_indices: Optional[set] = None,
    ) -> List[PseudoLabel]:
        """
        生成伪标签列表，自动排除金标注样本。

        Parameters
        ----------
        mecc_scores  : {sample_idx: (label, mecc_score, f_major)}
                       由 classify_vibration(return_score=True) 返回
                       mecc_score=1.0 / f_major=None 表示因 RMS 不足提前返回
        dl_probas    : {sample_idx: np.ndarray(num_classes)}
                       由 FullDatasetRunner.run_with_proba() 返回（面内方向）
        gold_indices : 已有金标注的样本索引集合，生成时自动排除

        Returns
        -------
        List[PseudoLabel]
        """
        gold_set    = gold_indices or set()
        all_indices = set(mecc_scores.keys()) | set(dl_probas.keys())
        pseudo_list: List[PseudoLabel] = []

        for idx in all_indices:
            if idx in gold_set:
                continue
            pl = self._classify_single(idx, mecc_scores.get(idx), dl_probas.get(idx))
            if pl is not None:
                pseudo_list.append(pl)

        n_viv = sum(p.class_id == _VIV_CLASS    for p in pseudo_list)
        n_nor = sum(p.class_id == _NORMAL_CLASS for p in pseudo_list)
        logger.info(
            f"伪标签生成完成：总计 {len(pseudo_list)} 条 | "
            f"VIV={n_viv} | Normal={n_nor}"
        )
        return pseudo_list

    # ------------------------------------------------------------------ #
    # 内部分类逻辑                                                          #
    # ------------------------------------------------------------------ #

    def _classify_single(
        self,
        idx:        int,
        mecc_entry: Optional[Tuple[int, float, Optional[float]]],
        dl_proba:   Optional[np.ndarray],
    ) -> Optional[PseudoLabel]:
        mecc_is_viv = mecc_is_nor = False
        mecc_conf   = 0.0

        if mecc_entry is not None:
            _, mecc_score, _ = mecc_entry
            if mecc_score is not None:
                if mecc_score < self.mecc_conf_viv:
                    mecc_is_viv = True
                    mecc_conf   = 1.0 - mecc_score
                elif mecc_score > self.mecc_conf_nor:
                    mecc_is_nor = True
                    mecc_conf   = mecc_score - self.mecc_conf_nor

        dl_is_viv = dl_is_nor = False
        dl_viv_p  = dl_nor_p  = 0.0

        if dl_proba is not None and len(dl_proba) > _VIV_CLASS:
            dl_viv_p = float(dl_proba[_VIV_CLASS])
            dl_nor_p = float(dl_proba[_NORMAL_CLASS])
            if dl_viv_p >= self.dl_conf_viv:
                dl_is_viv = True
            elif dl_nor_p >= self.dl_conf_nor:
                dl_is_nor = True

        if mecc_is_viv and dl_is_viv:
            return PseudoLabel(idx, _VIV_CLASS, "both", (mecc_conf + dl_viv_p) / 2)

        if mecc_is_nor and dl_is_nor:
            return PseudoLabel(idx, _NORMAL_CLASS, "both", (mecc_conf + dl_nor_p) / 2)

        if dl_is_viv and not mecc_is_viv:
            return PseudoLabel(idx, _VIV_CLASS, "dl", dl_viv_p)

        if mecc_is_viv and not dl_is_viv:
            return PseudoLabel(idx, _VIV_CLASS, "mecc", mecc_conf)

        if mecc_is_nor and not dl_is_nor:
            return PseudoLabel(idx, _NORMAL_CLASS, "mecc", mecc_conf)

        return None
