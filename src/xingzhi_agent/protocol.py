"""实验方案生成器 / dilution-series + controls protocol generator, stdlib only.

纯算术 + 模板：给定储备液浓度、目标浓度、终体积、重复数，
用 C1V1=C2V2 算出每一步的具体移液体积（0.1 µL 取整），
并给出多点稀释梯度、对照设置、读出方式与 Go/No-Go 判据。

不适用（诚实标注）：
- 不处理储备液本身的配制（如粉末称量、溶解度）；
- 不替代移液器量程核对：最高点移液体积 < 1 µL 时自动改走 100× 中间稀释并给出提示；
  梯度点体积 < 0.2 µL 时提示低于 P2 移液器量程下限；非零体积永不取整为 0.0
  （0.0 µL 会让方案看起来"不用加样"，是错的）；
- 浓度/体积单位只接受 nM/µM/mM/M 与 nL/µL/mL/L（含 µ 与 μ 两种写法），其余单位明确报错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Protocol", "design_protocol", "parse_concentration", "parse_volume"]

# 统一换算到 µM（浓度）/ µL（体积）
_CONC_UNITS = {"nm": 0.001, "µm": 1.0, "um": 1.0, "μm": 1.0, "mm": 1000.0, "m": 1_000_000.0}
_VOL_UNITS = {"nl": 0.001, "µl": 1.0, "ul": 1.0, "μl": 1.0, "ml": 1000.0, "l": 1_000_000.0}

_READOUTS = {
    "BLI": "BLI 结合曲线：KD、kon/koff；4PL 拟合 R² ≥ 0.95 方可采信",
    "SPR": "SPR 响应曲线：KD 拟合，chi² 小于最大响应值的 10%",
    "ELISA": "OD450 读数：信噪比（阳性/空白）≥ 10",
    "SEC-HPLC": "单体峰面积占比；聚集峰面积同步记录",
}


def _norm_unit(unit: str) -> str:
    return unit.strip().lower().replace("μ", "µ")


def parse_concentration(value: float | str, unit: str | None = None) -> float:
    """解析浓度 → µM。接受 100.0 + unit="µM"，或 "10 mM" 这样的字符串。"""
    if isinstance(value, str):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Zµμ]+)\s*", value)
        if not m:
            raise ValueError(f"无法解析浓度：{value!r}（示例：'10 mM' 或 100.0 + unit='µM'）")
        num, unit = float(m.group(1)), m.group(2)
    else:
        num = float(value)
    if unit is None:
        raise ValueError("浓度必须带单位（nM/µM/mM/M），如 unit='µM'")
    factor = _CONC_UNITS.get(_norm_unit(unit))
    if factor is None:
        raise ValueError(f"不支持的浓度单位：{unit!r}（仅支持 nM/µM/mM/M）")
    if not num > 0:
        raise ValueError(f"浓度必须为正数：{value!r}")
    return num * factor


def parse_volume(value: float | str, unit: str | None = None) -> float:
    """解析体积 → µL。接受 1.0 + unit="mL"，或 "200 µL" 这样的字符串。"""
    if isinstance(value, str):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Zµμ]+)\s*", value)
        if not m:
            raise ValueError(f"无法解析体积：{value!r}（示例：'1 mL' 或 200.0 + unit='µL'）")
        num, unit = float(m.group(1)), m.group(2)
    else:
        num = float(value)
    if unit is None:
        raise ValueError("体积必须带单位（nL/µL/mL/L），如 unit='µL'")
    factor = _VOL_UNITS.get(_norm_unit(unit))
    if factor is None:
        raise ValueError(f"不支持的体积单位：{unit!r}（仅支持 nL/µL/mL/L）")
    if not num > 0:
        raise ValueError(f"体积必须为正数：{value!r}")
    return num * factor


def _round01(x: float) -> float:
    return round(x, 1)


def _round_vol(x: float) -> float:
    """移液体积取整：≥0.1 µL 取整到 0.1；更小的值保留更多小数位，
    绝不把非零体积取整为 0.0（0.0 µL 会让方案看起来"不用加样"，是错的）。"""
    if x >= 0.1:
        return round(x, 1)
    if x >= 0.01:
        return round(x, 2)
    if x > 0:
        return round(x, 3)
    return 0.0


@dataclass
class Protocol:
    """一份实验方案（全部数值为 µM/µL，可直接 JSON 序列化）。"""

    stock_um: float
    target_um: float
    final_volume_ul: float
    replicates: int
    assay: str
    target_name: str
    series: list[dict[str, Any]] = field(default_factory=list)
    single_point: dict[str, Any] = field(default_factory=dict)
    intermediate: dict[str, Any] | None = None
    total_stock_ul: float = 0.0
    total_buffer_ul: float = 0.0
    controls: dict[str, str] = field(default_factory=dict)
    readout: str = ""
    go_no_go: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stock_um": self.stock_um,
            "target_um": self.target_um,
            "final_volume_ul": self.final_volume_ul,
            "replicates": self.replicates,
            "assay": self.assay,
            "target_name": self.target_name,
            "series": self.series,
            "single_point": self.single_point,
            "intermediate": self.intermediate,
            "total_stock_ul": self.total_stock_ul,
            "total_buffer_ul": self.total_buffer_ul,
            "controls": self.controls,
            "readout": self.readout,
            "go_no_go": self.go_no_go,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines = [
            f"## 实验方案：{self.target_name} · {self.assay}（{self.replicates} 重复）",
            "",
            f"- 储备液：{self.stock_um:.4g} µM；目标（梯度最高点）：{self.target_um:.4g} µM；"
            f"每孔终体积 {self.final_volume_ul:.4g} µL",
        ]
        if self.intermediate:
            it = self.intermediate
            lines.append(
                f"- 中间稀释（单点移液体积 < 1 µL，改走 100× 中间液）："
                f"{it['stock_ul']} µL 储备液 + {it['buffer_ul']} µL 缓冲液 → "
                f"{it['concentration_um']:.4g} µM 中间液 {it['prep_volume_ul']:.0f} µL；"
                f"后续各点按中间液移液"
            )
        lines += ["", "| 梯度点 | 浓度 (µM) | 储备液/中间液 (µL) | 缓冲液 (µL) |", "|---|---|---|---|"]
        for s in self.series:
            lines.append(
                f"| {s['point']} | {s['concentration_um']:.4g} | {s['stock_ul']} | {s['buffer_ul']} |"
            )
        lines += [
            "",
            f"- 用量合计（{self.replicates} 重复 × {len(self.series)} 点）："
            f"储备液/中间液 {self.total_stock_ul} µL，缓冲液 {self.total_buffer_ul} µL",
            "",
            "### 对照",
            f"- 阳性对照：{self.controls['positive']}",
            f"- 阴性对照：{self.controls['negative']}",
            f"- 空白：{self.controls['blank']}",
            "",
            f"### 读出\n\n{self.readout}",
            "",
            f"### Go / No-Go\n\n{self.go_no_go}",
        ]
        if self.warnings:
            lines += ["", "### 提示"] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def design_protocol(
    stock_concentration: float | str,
    target_concentration: float | str,
    final_volume: float | str,
    replicates: int = 3,
    *,
    stock_unit: str | None = None,
    target_unit: str | None = None,
    volume_unit: str | None = None,
    assay: str = "BLI",
    target_name: str = "EGFR",
    series_points: int = 5,
    series_fold: float = 3.0,
) -> Protocol:
    """生成稀释梯度实验方案。

    stock/target/final_volume 可传数值 + 对应单位参数，或直接传 "10 mM" / "1 mL" 字符串。
    目标浓度高于储备液浓度时抛 ValueError（无法稀释，不是静默截断）。
    """
    stock_um = parse_concentration(stock_concentration, stock_unit)
    target_um = parse_concentration(target_concentration, target_unit)
    final_ul = parse_volume(final_volume, volume_unit)
    if target_um > stock_um:
        raise ValueError(
            f"目标浓度 {target_um:.4g} µM 高于储备液 {stock_um:.4g} µM，无法稀释："
            "请提高储备液浓度或降低目标浓度"
        )
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError(f"重复数必须为 ≥1 的整数，当前 {replicates!r}")
    if series_points < 2:
        raise ValueError(f"梯度点数至少 2，当前 {series_points}")
    if series_fold <= 1:
        raise ValueError(f"梯度倍数必须 > 1，当前 {series_fold}")

    assay = (assay or "BLI").strip().upper()
    warnings: list[str] = []

    # 单点体积（最高点）：C1V1 = C2V2
    v_stock = target_um * final_ul / stock_um
    use_intermediate = v_stock < 1.0
    intermediate: dict[str, Any] | None = None
    source_um = stock_um
    source_label = "储备液"
    if use_intermediate:
        # 100× 中间液：10 µL 储备液 + 990 µL 缓冲液 → 1000 µL
        source_um = stock_um / 100.0
        intermediate = {
            "factor": 100,
            "concentration_um": round(source_um, 6),
            "prep_volume_ul": 1000.0,
            "stock_ul": 10.0,
            "buffer_ul": 990.0,
        }
        source_label = "中间液"
        warnings.append(
            f"最高点单点移液 {v_stock:.2f} µL < 1 µL，已改走 100× 中间液（{source_um:.4g} µM）"
        )
        v_stock = target_um * final_ul / source_um

    single_point = {
        "concentration_um": round(target_um, 6),
        "stock_ul": _round_vol(v_stock),
        "buffer_ul": _round01(final_ul - _round_vol(v_stock)),
        "source": source_label,
    }

    series: list[dict[str, Any]] = []
    for i in range(series_points):
        conc = target_um / (series_fold ** i)
        v = conc * final_ul / source_um
        if 0 < v < 0.2:
            warnings.append(
                f"梯度点 {i + 1}（{conc:.4g} µM）移液体积 {v:.3f} µL 低于 P2 移液器量程下限（0.2 µL）："
                "建议改走中间稀释或缩小梯度范围"
            )
        v_r = _round_vol(v)
        series.append({
            "point": i + 1,
            "concentration_um": round(conc, 6),
            "stock_ul": v_r,
            "buffer_ul": _round01(final_ul - v_r),
            "source": source_label,
        })

    total_stock = _round01(sum(s["stock_ul"] for s in series) * replicates)
    total_buffer = _round01(sum(s["buffer_ul"] for s in series) * replicates)

    return Protocol(
        stock_um=stock_um,
        target_um=target_um,
        final_volume_ul=final_ul,
        replicates=replicates,
        assay=assay,
        target_name=target_name,
        series=series,
        single_point=single_point,
        intermediate=intermediate,
        total_stock_ul=total_stock,
        total_buffer_ul=total_buffer,
        controls={
            "positive": f"已知结合 {target_name} 的参比纳米抗体，{target_um:.4g} µM，{assay} 读出应为阳性",
            "negative": f"无关靶点纳米抗体（同浓度）+ 无靶点空白传感器/孔，{assay} 读出应接近本底",
            "blank": "仅缓冲液，不加样品与靶点，用于扣除本底漂移",
        },
        readout=_READOUTS.get(assay, f"{assay} 读出：按所选方法记录响应值并做 4PL 拟合"),
        go_no_go=(
            f"进入下一轮的判据：{assay} 测得 KD < 100 nM（对 {target_name}），"
            f"且阳性对照信号 ≥ 阴性对照 10 倍；任一不满足则回到序列层面修复，"
            "不扩大筛选规模。"
        ),
        warnings=warnings,
    )
