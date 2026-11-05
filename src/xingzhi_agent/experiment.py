"""实验设计建议器 / Next-experiment designer.

创新点 ③：不是泛泛的"建议做实验"，而是依据本轮理化指标与文献命中，
给出带条件、对照、样本量与**判别价值**（期望信息增益粗估）的下一步实验方案。
"""

from __future__ import annotations

from typing import Any

from .physchem import analyze


def _risk_rank(report: dict[str, Any], assay_ph: float) -> list[dict[str, Any]]:
    """把理化风险排成优先级队列：越不确定、越便宜验证的越靠前。"""
    m = {x["name"]: x for x in report["metrics"]}
    risks: list[dict[str, Any]] = []

    pi = float(m["等电点 pI"]["value"])
    if abs(pi - assay_ph) < 1.5:
        risks.append({
            "risk": f"pI {pi} 距测定 pH {assay_ph} 仅 {abs(pi - assay_ph):.2f}，净电荷接近零，溶解度/聚集风险高",
            "experiment": f"在 pH {assay_ph} 与 pH {round(pi + 2.0, 1)} 两种缓冲下做 SEC-HPLC 单体率对比（各 n=3）",
            "controls": "阳性对照：已知可溶 VHH（如抗 GFP 纳米抗体）；阴性对照：热变性样品",
            "readout": "单体峰面积占比 / 聚集峰面积",
            "discriminative_value": "高：若两 pH 单体率差 >15%，直接判定是电荷驱动聚集，可改走缓冲优化而非序列重设计",
            "cost": "低（1 天，~¥500 耗材）",
        })

    patch = float(m["最大疏水补丁(9aa)"]["value"])
    if patch > 1.0:
        risks.append({
            "risk": f"最大疏水窗口 {patch}，非特异结合与聚集风险",
            "experiment": "Poly-specificity reagent (PSR) 流式打分 + 基线 BLI 非特异结合（n=2 重复）",
            "controls": "高非特异对照抗体 + 低非特异临床抗体各一",
            "readout": "PSR 归一化 MFI；BLI 对空白传感器的漂移",
            "discriminative_value": "中高：PSR 分数 >0.3 则优先做疏水补丁回突变，而不是继续优化亲和力",
            "cost": "低（2 天）",
        })

    cys = float(m["半胱氨酸 Cys"]["value"])
    if cys > 2 or cys % 2 == 1:
        risks.append({
            "risk": f"Cys 数 {int(cys)}，可能存在游离巯基与错配二硫键",
            "experiment": "非还原 vs 还原 SDS-PAGE + Ellman 游离巯基定量",
            "controls": "还原剂处理组作为配对对照",
            "readout": "二聚体条带比例；游离 -SH 摩尔比",
            "discriminative_value": "高：若游离 -SH > 0.2 当量，直接做 C→S 突变而非继续表达优化",
            "cost": "极低（半天）",
        })

    ngly = float(m["N-糖基化基序 N-X-S/T"]["value"])
    if ngly > 0:
        risks.append({
            "risk": f"存在 {int(ngly)} 处 N-X-S/T 基序，真核表达可能被糖基化并遮蔽界面",
            "experiment": "大肠杆菌（无糖基化）与哺乳动物瞬转两种体系平行表达，PNGase F 处理对照",
            "controls": "PNGase F 处理组 / 未处理组",
            "readout": "SDS-PAGE 迁移率位移；两体系 KD 比值",
            "discriminative_value": "高：若两体系 KD 差 >3 倍，说明糖基化影响界面，须做 N→Q 突变",
            "cost": "中（1 周）",
        })

    if not risks:
        risks.append({
            "risk": "理化指标全部通过，主要不确定性在结合而非可表达性",
            "experiment": f"小规模表达（50 mL）后 BLI 亲和力测定，pH {assay_ph} 与 pH 7.4 双条件各 5 个浓度梯度",
            "controls": "无关靶点作阴性对照；已知结合剂作阳性对照",
            "readout": "KD、kon/koff；两 pH 下 KD 比值",
            "discriminative_value": "高：直接给出是否进入下一轮设计的 go/no-go 判据（KD < 100 nM）",
            "cost": "中（1–2 周）",
        })
    return risks


def design_experiment(
    sequence: str,
    molecule_type: str = "VHH",
    assay_ph: float = 6.0,
    target: str = "EGFR",
) -> dict[str, Any]:
    rep = analyze(sequence, molecule_type, assay_ph).as_dict()
    if not rep["ok"]:
        # S10：理化校验未通过（未知类型/空序列/非法字母等）时返回明确错误，
        # 不再让 _risk_rank 在空 metrics 上 KeyError 崩溃。
        return {
            "target": target,
            "assay_ph": assay_ph,
            "physchem_ok": False,
            "error": "；".join(rep["issues"]),
            "ranked_experiments": [],
            "go_no_go": "—",
            "sample_size_note": "理化校验未通过，未生成实验设计。",
        }
    plan = _risk_rank(rep, assay_ph)
    go_no_go = (
        f"进入下一轮的判据：pH {assay_ph} 下 SEC 单体率 ≥ 90%、PSR 非特异分数 < 0.3、"
        f"BLI KD < 100 nM（对 {target}），任一不满足则回到序列层面修复而非扩大筛选规模。"
    )
    return {
        "target": target,
        "assay_ph": assay_ph,
        "physchem_ok": rep["ok"],
        "ranked_experiments": plan,
        "go_no_go": go_no_go,
        "sample_size_note": "每条件 n≥3 生物学重复；先做最便宜且判别力最高的那一项（队列第 1 项）。",
    }
