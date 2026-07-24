"""td_copula 流水线：sequence | fit | sample | run | intra(口子)。"""

from __future__ import annotations

from pathlib import Path

from src.chapter4_characteristics.settings import load_config
from src.chapter4_characteristics.statistics.td_copula.fit import fit_class_td
from src.chapter4_characteristics.statistics.td_copula.sample import run_sample_class
from src.chapter4_characteristics.statistics.td_copula.sequence import extract_class_sequence


def _parse_class_ids(class_id: str | int) -> list[int]:
    if isinstance(class_id, int):
        return [class_id]
    text = str(class_id).strip().lower()
    if text == "all":
        return [0, 1, 2, 3]
    return [int(text)]


def run_sequence(
    class_id: str | int = "all",
    config_path: str | None = None,
    refresh: bool = False,
    max_samples: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for cid in _parse_class_ids(class_id):
        paths.append(
            extract_class_sequence(
                cid,
                config_path=config_path,
                refresh=refresh,
                max_samples=max_samples,
            )
        )
    return paths


def run_fit(
    class_id: str | int = "all",
    config_path: str | None = None,
) -> list[dict]:
    results: list[dict] = []
    for cid in _parse_class_ids(class_id):
        results.append(fit_class_td(cid, config_path=config_path))
    return results


def run_sample(
    class_id: str | int = "all",
    config_path: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for cid in _parse_class_ids(class_id):
        paths.append(
            run_sample_class(
                cid,
                config_path=config_path,
                n_paths=n_paths,
                n_steps=n_steps,
            )
        )
    return paths


def run_intra(class_id: str | int = "all", config_path: str | None = None) -> None:
    """
    口子：窗内过程时间相依 → 雨流计数对齐。

    本阶段不实现。后续可接入：
      - 谱特征 + 相位随机合成 σ(t)；或
      - 短时峰过程 Copula。
    """
    _ = load_config(config_path)
    ids = _parse_class_ids(class_id)
    raise NotImplementedError(
        "td_copula intra（窗内过程 / 雨流）尚未实现。"
        f"已预留 class_ids={ids}。"
        "当前可用：sequence → fit → sample（窗间 Markov + MC 轨迹）。"
    )


def run_td_full(
    class_id: str | int = "all",
    config_path: str | None = None,
    refresh: bool = False,
    max_samples: int | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
) -> dict:
    seq_paths = run_sequence(
        class_id=class_id,
        config_path=config_path,
        refresh=refresh,
        max_samples=max_samples,
    )
    fits = run_fit(class_id=class_id, config_path=config_path)
    sample_paths = run_sample(
        class_id=class_id,
        config_path=config_path,
        n_paths=n_paths,
        n_steps=n_steps,
    )
    return {
        "stage": "run",
        "sequence_paths": [str(p) for p in seq_paths],
        "fit_summary": [
            {
                "class_id": r["class_id"],
                "best_copula_type": r["best_copula_type"],
                "n_pairs": r["n_pairs"],
            }
            for r in fits
        ],
        "sample_paths": [str(p) for p in sample_paths],
    }


def run_td_copula_job(
    class_id: str | int = "all",
    config_path: str | None = None,
    stage: str = "run",
    refresh: bool = False,
    max_samples: int | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
) -> dict:
    stage = str(stage).strip().lower()
    if stage == "sequence":
        paths = run_sequence(
            class_id=class_id,
            config_path=config_path,
            refresh=refresh,
            max_samples=max_samples,
        )
        return {"stage": stage, "paths": [str(p) for p in paths]}
    if stage == "fit":
        results = run_fit(class_id=class_id, config_path=config_path)
        return {
            "stage": stage,
            "results": [
                {
                    "class_id": r["class_id"],
                    "best_copula_type": r["best_copula_type"],
                    "n_pairs": r["n_pairs"],
                }
                for r in results
            ],
        }
    if stage == "sample":
        paths = run_sample(
            class_id=class_id,
            config_path=config_path,
            n_paths=n_paths,
            n_steps=n_steps,
        )
        return {"stage": stage, "paths": [str(p) for p in paths]}
    if stage == "intra":
        run_intra(class_id=class_id, config_path=config_path)
        return {"stage": stage}
    if stage == "run":
        return run_td_full(
            class_id=class_id,
            config_path=config_path,
            refresh=refresh,
            max_samples=max_samples,
            n_paths=n_paths,
            n_steps=n_steps,
        )
    raise ValueError(
        f"未知 td_copula stage: {stage}；"
        "可选 sequence | fit | sample | run | intra"
    )
