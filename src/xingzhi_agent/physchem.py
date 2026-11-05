"""理化指标 / Physicochemical metrics, implemented from first principles (stdlib only).

不依赖 biopython 等重依赖，直接实现 pI / GRAVY / 指定 pH 净电荷 /
摩尔消光系数 / Cys 与 N-糖基化基序扫描 / CDR3 粗定位 / 最大疏水补丁，
每项都有阈值，可被校验器引用。

参考常数来源（公开教科书数值）：
- Kyte & Doolittle (1982) 疏水性标度 → GRAVY
- Bjellqvist et al. (1993/1994) pK 标度 → pI / 净电荷
- Pace et al. (1995) 消光系数 → A280

说明：本模块未实现 Guruprasad 不稳定性指数（DIWV）；勿在文档中宣称已支持。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Bjellqvist pK 值
PK_NTERM, PK_CTERM = 9.69, 2.34
PK_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
NEGATIVE = ("D", "E", "C", "Y")
POSITIVE = ("H", "K", "R")
MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16, "Q": 146.15,
    "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23,
    "Y": 181.19, "V": 117.15,
}
WATER = 18.015
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
NGLYC_RE = re.compile(r"(?=(N[^P][ST]))")  # 前瞻扫描：重叠位点（如 NNST 含 NNS 与 NST）各计一次（S9）
# VHH CDR3：FR3 末端保守 YYC 到 FR4 起始 WG.G 之间（IMGT 近似）。
# 锚定 YYC（O1）：旧正则只锚定 C，前面的 Cys 若在 30 aa 内遇到 WGQG 会误定位 CDR3 起点。
CDR3_RE = re.compile(r"YYC(?:AR|AK|AA|A|VR|TR)?(.{3,30}?)(?:W[GA][QKR]G)")


def net_charge(seq: str, ph: float) -> float:
    """Henderson–Hasselbalch 净电荷 / net charge at a given pH."""
    charge = 1.0 / (1.0 + 10 ** (ph - PK_NTERM)) - 1.0 / (1.0 + 10 ** (PK_CTERM - ph))
    for aa in POSITIVE:
        charge += seq.count(aa) / (1.0 + 10 ** (ph - PK_SIDE[aa]))
    for aa in NEGATIVE:
        charge -= seq.count(aa) / (1.0 + 10 ** (PK_SIDE[aa] - ph))
    return charge


def isoelectric_point(seq: str) -> float:
    """二分法求 pI / bisection search for the isoelectric point."""
    lo, hi = 0.0, 14.0
    # O3：60 轮已远超所需（约 18 轮即达 1e-4 精度），区间收窄到 1e-4 即提前退出
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return round((lo + hi) / 2, 2)


def gravy(seq: str) -> float:
    return round(sum(KD[a] for a in seq) / len(seq), 3) if seq else 0.0


def molecular_weight(seq: str) -> float:
    return round(sum(MW[a] for a in seq) - WATER * (len(seq) - 1), 1) if seq else 0.0


def extinction_280(seq: str) -> int:
    """Pace et al. 1995：E = 5500*W + 1490*Y + 125*C(假设全部成对)。"""
    return 5500 * seq.count("W") + 1490 * seq.count("Y") + 125 * (seq.count("C") // 2)


def extinction_280_reduced(seq: str) -> int:
    """还原值：全部 Cys 按游离巯基计（O2）。奇数 Cys 时真实 ε280 介于还原值与氧化值之间。"""
    return 5500 * seq.count("W") + 1490 * seq.count("Y") + 125 * seq.count("C")


def aromaticity(seq: str) -> float:
    return round(sum(seq.count(a) for a in "FWY") / len(seq), 3) if seq else 0.0


def hydrophobic_fraction(seq: str) -> float:
    return round(sum(seq.count(a) for a in "AVILMFWC") / len(seq), 3) if seq else 0.0


def max_hydrophobic_patch(seq: str, window: int = 9) -> float:
    """滑窗最大平均疏水性，粗略代理聚集倾向 / aggregation proxy."""
    if len(seq) < window:
        return gravy(seq)
    best = -9.9
    for i in range(len(seq) - window + 1):
        w = seq[i : i + window]
        best = max(best, sum(KD[a] for a in w) / window)
    return round(best, 3)


def cdr3(seq: str) -> tuple[str, int]:
    m = CDR3_RE.search(seq)
    return (m.group(1), len(m.group(1))) if m else ("", 0)


@dataclass
class Metric:
    name: str
    value: Any
    unit: str = ""
    ok: bool | None = None
    threshold: str = ""
    basis: str = ""  # 阈值依据 / rationale


@dataclass
class PhysChemReport:
    ok: bool
    metrics: list[Metric] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": self.issues,
            "flags": self.flags,
            "metrics": [m.__dict__ for m in self.metrics],
        }


LENGTH_RULES = {"VHH": (100, 150), "MINIBINDER": (40, 100)}


def analyze(seq: str, molecule_type: str = "VHH", assay_ph: float = 6.0) -> PhysChemReport:
    """返回带阈值判定的理化报告 / full physicochemical report with pass-fail calls."""
    s = re.sub(r"\s+", "", seq or "").upper()
    # S10：未知类型直接报错，不再静默按 VHH 处理
    mtype_raw = (molecule_type or "").strip().upper()
    if mtype_raw not in {"VHH", "MINIBINDER", "MINI", "MB"}:
        rep = PhysChemReport(ok=False)
        rep.issues.append(f"未知 molecule_type: {molecule_type!r}（仅支持 VHH/minibinder）")
        return rep
    mtype = "MINIBINDER" if mtype_raw in {"MINIBINDER", "MINI", "MB"} else "VHH"
    rep = PhysChemReport(ok=True)

    if not s:
        rep.ok = False
        rep.issues.append("序列为空")
        return rep
    if not AA_RE.match(s):
        rep.ok = False
        bad = sorted(set(s) - set(AA_RE.pattern[2:-3]))
        rep.issues.append(f"含非标准氨基酸字母：{''.join(bad)}（仅允许 20 种标准 AA）")
        return rep

    lo, hi = LENGTH_RULES[mtype]
    length_ok = lo <= len(s) <= hi
    rep.metrics.append(Metric("长度 length", len(s), "aa", length_ok, f"{lo}–{hi}",
                              "VHH 成熟肽常见 100–150 aa；de novo minibinder 常见 40–100 aa"))
    if not length_ok:
        rep.ok = False
        rep.issues.append(f"长度 {len(s)} 超出 {mtype} 区间 [{lo}, {hi}]")

    pi = isoelectric_point(s)
    q_assay = round(net_charge(s, assay_ph), 2)
    q74 = round(net_charge(s, 7.4), 2)
    g = gravy(s)
    patch = max_hydrophobic_patch(s)
    cys = s.count("C")
    ngly = [m.start() + 1 for m in NGLYC_RE.finditer(s)]
    _, cdr3_len = cdr3(s)

    pi_ok = not (assay_ph - 0.5 <= pi <= assay_ph + 0.5)
    rep.metrics.append(Metric("等电点 pI", pi, "", pi_ok, f"远离测定 pH {assay_ph}±0.5",
                              "pI 落在测定缓冲 pH 附近时净电荷≈0，溶解度最低、易聚集"))
    if not pi_ok:
        rep.flags.append(f"pI {pi} 距测定 pH {assay_ph} 过近，溶解度风险升高")

    rep.metrics.append(Metric(f"净电荷 @pH {assay_ph}", q_assay, "e", abs(q_assay) >= 1.0, "|q| ≥ 1",
                              "测定条件下保持一定净电荷有助于胶体稳定性"))
    rep.metrics.append(Metric("净电荷 @pH 7.4", q74, "e", None, "参考值", "生理 pH 下的电荷参照"))

    gravy_ok = g <= 0.0
    rep.metrics.append(Metric("GRAVY", g, "", gravy_ok, "≤ 0",
                              "Kyte–Doolittle 平均疏水性；可溶球状蛋白通常为负"))
    if not gravy_ok:
        rep.flags.append(f"GRAVY {g} > 0，整体偏疏水，表达/溶解风险升高")

    patch_ok = patch <= 1.8
    rep.metrics.append(Metric("最大疏水补丁(9aa)", patch, "", patch_ok, "≤ 1.8",
                              "连续疏水窗口是聚集与非特异结合的主要代理指标"))
    if not patch_ok:
        rep.flags.append(f"存在强疏水窗口（{patch}），建议做非特异结合筛查")

    cys_ok = cys <= 4
    rep.metrics.append(Metric("半胱氨酸 Cys", cys, "个", cys_ok, "≤ 4（经典 VHH 为 2）",
                              "游离 Cys 易致二聚与批次不稳；额外 Cys 需人工复核"))
    if not cys_ok:
        rep.flags.append(f"Cys 共 {cys} 个，请检查二硫键配对与游离巯基")
    if cys % 2 == 1:
        rep.flags.append("Cys 为奇数，存在游离巯基风险")

    ngly_ok = not ngly
    rep.metrics.append(Metric("N-糖基化基序 N-X-S/T", len(ngly), "处", ngly_ok, "界面区应为 0",
                              "真核表达体系中 N-X-S/T 可被糖基化，可能遮蔽结合界面"))
    if ngly:
        rep.flags.append(f"N-X-S/T 基序位置：{ngly}（界面区建议突变消除）")

    if mtype == "VHH":
        cdr3_ok = 8 <= cdr3_len <= 24 if cdr3_len else None
        rep.metrics.append(Metric("CDR3 长度(估计)", cdr3_len or "未识别", "aa", cdr3_ok, "8–24",
                                  "骆驼科天然库 CDR3 偏长，过短难覆盖凹陷表位、过长熵损失大"))
        if cdr3_len and not cdr3_ok:
            rep.flags.append(f"CDR3 估计长度 {cdr3_len} 超出常见区间 8–24")
        if not cdr3_len:
            # O1：CDR3 未识别时不再静默中性，明确告警人工复核
            rep.flags.append("CDR3 未识别（非典型 FR4 或超长 CDR3），请人工复核")

    rep.metrics.append(Metric("分子量 MW", molecular_weight(s), "Da", None, "", "用于纯化与质谱确认"))
    rep.metrics.append(Metric("ε280", extinction_280(s), "M⁻¹cm⁻¹", None, "",
                              "A280 定量浓度所需"))
    if cys % 2 == 1:
        # O2：奇数 Cys 时同时给出还原值（全部 Cys 按游离巯基计），真实值介于两者之间；
        # 偶数 Cys 时氧化值即足够，不新增指标，避免离线笔记无故变化。
        rep.metrics.append(Metric("ε280（还原）", extinction_280_reduced(s), "M⁻¹cm⁻¹", None, "",
                                  "全部 Cys 按游离巯基计；奇数 Cys 时真实值介于还原值与氧化值之间"))
    rep.metrics.append(Metric("芳香族比例", aromaticity(s), "", None, "", "影响 A280 与界面堆积"))
    rep.metrics.append(Metric("疏水残基比例", hydrophobic_fraction(s), "", None, "", "整体疏水负荷参考"))
    return rep
