"""Copula 探索图统一标注：论文符号 + 面内/面外配色。"""

from __future__ import annotations

from src.figure_paintings.figs_for_thesis.config import VIV_OUTPLANE_COLOR, get_blue_color_map

# 面内蓝、面外淡红
INPLANE_COLOR = get_blue_color_map(style="discrete", start_map_index=1, end_map_index=4).colors[1]
OUTPLANE_COLOR = "#E88B86"  # 淡于 VIV_OUTPLANE_COLOR，直方图更柔和
OUTPLANE_EDGE = VIV_OUTPLANE_COLOR


def parse_var_meta(var_name: str) -> tuple[str, str, int]:
    """
    freq_in_3 → ('freq', 'in', 3)
    energy_out_12 → ('energy', 'out', 12)
    """
    parts = var_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"无法解析变量名：{var_name}")
    kind = parts[0]  # freq / energy
    plane = parts[1]  # in / out
    idx = int(parts[2])
    return kind, plane, idx


def paper_label(var_name: str, mathtext: bool = True) -> str:
    """程序名 → 论文符号：f_k / E_k（面外加 ⊥）。"""
    kind, plane, idx = parse_var_meta(var_name)
    base = "f" if kind == "freq" else "E"
    if mathtext:
        if plane == "in":
            return rf"${base}_{{{idx}}}$"
        return rf"${base}_{{{idx}}}^{{\perp}}$"
    if plane == "in":
        return f"{base}_{idx}"
    return f"{base}_{idx}^⊥"


def paper_pair_label(name_i: str, name_j: str) -> str:
    return f"{paper_label(name_i)}–{paper_label(name_j)}"


def plane_color(var_name: str) -> str:
    _, plane, _ = parse_var_meta(var_name)
    return INPLANE_COLOR if plane == "in" else OUTPLANE_COLOR


def is_inplane(var_name: str) -> bool:
    return parse_var_meta(var_name)[1] == "in"


def short_form(form: str | None) -> str:
    if not form:
        return "?"
    if form.startswith("gmm_"):
        return f"GMM{form.split('_', 1)[1]}"
    mapping = {
        "gamma": "Γ",
        "lognorm": "LN",
        "norm": "N",
        "expon": "Exp",
    }
    return mapping.get(form, form)


def paper_labels_for_list(var_names: list[str]) -> list[str]:
    return [paper_label(n) for n in var_names]
