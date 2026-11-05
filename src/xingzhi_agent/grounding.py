"""可验证引用闭环 / Claim-level citation grounding.

创新点 ①：把笔记拆成"主张（claim）"，每条主张必须携带本轮检索命中的文献 ID，
并与该文献的标题 + 摘要做词级重叠（span matching）打分；无引用或引用与证据不匹配
的主张会被标为未接地（ungrounded），据此给出 claim-level grounding score。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .figures import FIG_CITE_RE, figure_caption

CITE_RE = re.compile(r"\[(P\d{3}(?:\s*[,;，、]\s*P\d{3})*)\]")
TOOL_CITE_RE = re.compile(r"\[TOOL:([a-z_]+)\]")
ID_RE = re.compile(r"P\d{3}")
STOP = set(
    "的 了 和 与 在 是 为 对 及 等 中 上 下 并 可 将 该 本 其 以 由 也 都 一 个 这 那 使 用 "
    "a an the and or of to in on for with by is are be as that this from at we our it "
    "no up do go us vs et al de".split()
)
# 需要证据支撑的"事实性主张"信号词：含数字、比较级、机制断言
FACT_HINTS = ("提高", "降低", "优于", "相关", "导致", "显著", "建议", "表明", "证明", "增加", "减少", "影响")
# 这些行不算主张（标题、表头、分隔、声明、空行）
SKIP_PREFIX = ("#", "|", ">", "---", "```", "<!--")


@dataclass
class Claim:
    text: str
    cites: list[str] = field(default_factory=list)
    tool_cites: list[str] = field(default_factory=list)
    # P2-1：图表引用 [Pxxx:FigN]，(paper_id, fig_no)；图注即证据文本
    fig_cites: list[tuple[str, int]] = field(default_factory=list)
    needs_evidence: bool = True
    grounded: bool = False
    overlap: float = 0.0
    matched_id: str = ""
    reason: str = ""



# 中英术语桥：语料多为英文摘要、笔记多为中文，纯词面重叠会落空。
GLOSSARY = {
    "纳米抗体": ["nanobody", "nanobodies", "vhh"], "单域抗体": ["single-domain", "sdab", "vhh"],
    "抗体": ["antibody", "antibodies"], "抗原": ["antigen"], "表位": ["epitope"],
    "亲和力": ["affinity", "binding"], "结合": ["binding", "bind"], "溶解度": ["solubility", "soluble"],
    "稳定性": ["stability", "stable"], "表达": ["expression", "expressed", "express"],
    "纯化": ["purification", "purify"], "大肠杆菌": ["coli", "escherichia"],
    "膜蛋白": ["membrane"], "构象": ["conformation", "conformational"],
    "结构": ["structure", "structural"], "预测": ["prediction", "predict", "predicted"],
    "筛选": ["screening", "screen", "selection"], "文库": ["library", "repertoire"],
    "序列": ["sequence"], "长度": ["length"], "疏水": ["hydrophobic", "hydrophobicity", "gravy"],
    "电荷": ["charge"], "等电点": ["isoelectric", "pi"], "糖基化": ["glycosylation"],
    "二硫键": ["disulfide"], "热稳定": ["thermostability", "thermal"],
    "免疫": ["immune", "immunization", "immunized"], "骆驼": ["camelid", "camel"],
    "重链": ["heavy-chain", "heavy"], "轻链": ["light-chain", "light"],
    "结构域": ["domain"], "多样性": ["diversity", "diverse"], "实验": ["experiment", "experimental", "vitro", "vivo"],
    "计算": ["computational", "silico"], "模型": ["model", "models"], "机器学习": ["learning", "deep"],
    "设计": ["design", "designed"], "优化": ["optimization", "optimize"], "工程": ["engineering", "engineered"],
    "亲水": ["hydrophilic"], "聚集": ["aggregation", "aggregate"], "产量": ["yield"],
    "治疗": ["therapeutic", "therapy"], "肿瘤": ["tumor", "cancer"], "靶点": ["target"],
    "片段": ["fragment"], "突变": ["mutation", "mutant"], "亲和成熟": ["maturation"],
}


def _bridge(text: str, toks: set[str]) -> set[str]:
    """把中文术语补成英文同义词，使中文笔记能与英文摘要对齐。"""
    out = set(toks)
    for cn, ens in GLOSSARY.items():
        if cn in text:
            out.update(ens)
    return out


# ---------- 方向判定 / direction words ----------
# 主张与证据的方向词表：用于发现"主张说提高、文献却说降低"这类方向相反的引用。
UP_ZH = ("提高", "升高", "增加", "优于", "增强", "提升", "改善", "上升")
DOWN_ZH = ("降低", "下降", "减少", "削弱", "劣于", "减弱")
UP_EN = {
    "increase", "increases", "increased", "increasing", "improve", "improves",
    "improved", "improving", "higher", "highest", "enhance", "enhances",
    "enhanced", "enhancing", "rise", "rises", "rising", "rose", "risen",
    "raise", "raises", "raised", "raising",
    "boost", "boosts", "boosted", "upregulate", "upregulates", "upregulated",
    "upregulation",
}
DOWN_EN = {
    "decrease", "decreases", "decreased", "decreasing", "reduce", "reduces",
    "reduced", "reducing", "lower", "lowers", "lowered", "lowering", "impair",
    "impairs", "impaired", "impairment", "drop", "drops", "dropped", "dropping",
    "decline", "declines", "declined", "declining", "downregulate", "downregulates",
    "downregulated", "downregulation", "fall", "falls", "fell", "fallen",
}


def direction_of(text: str) -> str:
    """返回 up / down / mixed / neutral。英文按整词匹配，避免 rise 误中 arise。"""
    low = text.lower()
    en_toks = set(re.findall(r"[a-z][a-z\-]{2,}", low))
    up = sum(1 for w in UP_ZH if w in text) + sum(1 for w in en_toks if w in UP_EN)
    down = sum(1 for w in DOWN_ZH if w in text) + sum(1 for w in en_toks if w in DOWN_EN)
    if up and down:
        return "mixed"
    if up > down:
        return "up"
    if down > up:
        return "down"
    return "neutral"


def directions_conflict(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    """仅当主张与证据方向严格相反（一方纯上、另一方纯下）时判冲突。

    保守口径：证据本身含双向表述（mixed）时不判冲突；主张本身是 mixed 时也不判。
    这样做是因为证据文本常把主张原句也包含在内，宽松口径会把合法引用误判为冲突。
    """
    cd = direction_of(claim_text)
    if cd not in ("up", "down"):
        return False, ""
    low = evidence_text.lower()
    en_toks = set(re.findall(r"[a-z][a-z\-]{2,}", low))
    ev_up = sum(1 for w in UP_ZH if w in evidence_text) + sum(1 for w in en_toks if w in UP_EN)
    ev_down = sum(1 for w in DOWN_ZH if w in evidence_text) + sum(1 for w in en_toks if w in DOWN_EN)
    if cd == "up" and ev_down > 0 and ev_up == 0:
        return True, "主张方向为提高/增加，但引用文献方向为降低/减少，结论相反"
    if cd == "down" and ev_up > 0 and ev_down == 0:
        return True, "主张方向为降低/减少，但引用文献方向为提高/增加，结论相反"
    return False, ""


def overlap_threshold(n_tokens: int) -> float:
    """按主张词元数自适应：短主张只凭一两个通用词就能过线，必须要求更高重叠。"""
    if n_tokens <= 4:
        return 0.30
    if n_tokens <= 8:
        return 0.20
    return 0.12


# ---------- 工具徽章数值核对 / tool-claim numeric verification ----------

# 主张文本里的理化数值抽取：(正则, 指标名, 容差)。指标名与 physchem metrics 的 name 对应。
# _SEP 容忍 "=", "≈", "："等分隔写法；kDa 先换算成 Da 再比对（容差 100 Da）。
_SEP = r"[^\d+\-]{0,6}?"
_SEQ_NUM_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"pI" + _SEP + r"([+-]?\d+(?:\.\d+)?)"), "等电点 pI", 0.05),
    (re.compile(r"等电点" + _SEP + r"([+-]?\d+(?:\.\d+)?)"), "等电点 pI", 0.05),
    (re.compile(r"长度" + _SEP + r"(\d+)\s*aa"), "length", 0),
    (re.compile(r"GRAVY" + _SEP + r"([+-]?\d+(?:\.\d+)?)"), "GRAVY", 0.05),
    (re.compile(r"(?:分子量|MW)" + _SEP + r"([\d,]+(?:\.\d+)?)\s*kDa"), "分子量 MW", 100.0),
    (re.compile(r"(?:分子量|MW)" + _SEP + r"([\d,]+(?:\.\d+)?)\s*Da"), "分子量 MW", 1.0),
    (re.compile(r"最大疏水(?:窗口|补丁)(?:[（(]\s*9\s*aa\s*[）)])?" + _SEP + r"([+-]?\d+(?:\.\d+)?)"), "最大疏水补丁(9aa)", 0.05),
    (re.compile(r"(?:半胱氨酸(?:\s*Cys)?|Cys)" + _SEP + r"(\d+)\s*个"), "半胱氨酸 Cys", 0),
    (re.compile(r"N-糖基化(?:基序)?(?:\s*N-X-S/T)?" + _SEP + r"(\d+)\s*处"), "N-糖基化基序 N-X-S/T", 0),
    (re.compile(r"CDR3[^\d]{0,12}?(\d+)\s*aa"), "CDR3 长度(估计)", 0),
    (re.compile(r"ε280" + _SEP + r"([\d,]+)"), "ε280", 1),
]
_CHARGE_RE = re.compile(r"净电荷\s*(?:@\s*pH\s*(\d+(?:\.\d+)?))?\s*[为是:=]?\s*([+-]?\d+(?:\.\d+)?)")


def _charge_metric_key(text: str, seq_metrics: dict[str, Any]) -> str | None:
    """按主张提到的 pH（没有则用测定 pH）找到对应的净电荷指标名。"""
    ph = None
    m = re.search(r"pH\s*(\d+(?:\.\d+)?)", text)
    if m:
        ph = float(m.group(1))
    elif seq_metrics.get("assay_ph") is not None:
        try:
            ph = float(seq_metrics["assay_ph"])
        except (TypeError, ValueError):
            ph = None
    if ph is None:
        return None
    for key in seq_metrics:
        km = re.fullmatch(r"净电荷 @pH ([\d.]+)", key)
        if km and abs(float(km.group(1)) - ph) <= 0.05:
            return key
    return None


def check_sequence_numbers(text: str, seq_metrics: dict[str, Any] | None) -> tuple[bool, str]:
    """核对主张中的理化数值与本轮 sequence_check 输出。返回 (通过, 原因)。

    无数值可核对时返回通过（无可证伪内容）；有数值但未传入计算结果时返回不通过。
    """
    found: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for pat, key, tol in _SEQ_NUM_PATTERNS:
        m = pat.search(text)
        if m and key not in seen:
            if key == "length" and "CDR3" in text[max(0, m.start() - 12):m.start()]:
                continue  # "CDR3 估计长度 5 aa"里的长度是 CDR3 长度，不是序列长度
            seen.add(key)
            val = float(m.group(1).replace(",", ""))
            if "kDa" in m.group(0):
                val *= 1000.0  # kDa 写法先换算成 Da 再比对
            found.append((key, val, tol))
    cm = _CHARGE_RE.search(text)
    if cm:
        val = float(cm.group(2))
        key = _charge_metric_key(text, seq_metrics or {})
        if key is not None:
            found.append((key, val, 0.05))
        # 主张未指明可核对的 pH、且本轮也没有该 pH 的指标时，跳过该项不判
    if not found:
        return True, ""
    if seq_metrics is None:
        return False, "主张含理化数值，但未传入本轮理化计算结果，无法核对"
    bad: list[str] = []
    for key, val, tol in found:
        expect = seq_metrics.get(key)
        if expect is None or isinstance(expect, str):
            continue  # 本轮未输出该指标（如 CDR3 未识别），跳过
        try:
            exp = float(expect)
        except (TypeError, ValueError):
            continue
        if abs(val - exp) > tol:
            bad.append(f"{key} 主张 {val:g} 与实测 {exp:g} 不符")
    if bad:
        return False, "；".join(bad)
    return True, ""


_EXP_MARKERS = (
    "SEC-HPLC", "PSR", "BLI", "SDS-PAGE", "PNGase", "Ellman",
    "单体率", "游离巯基", "糖基化", "判据", "KD", "单体峰", "非特异",
)


def _plan_text(plan: dict[str, Any]) -> str:
    parts: list[str] = []
    for e in plan.get("ranked_experiments", []) or []:
        for k in ("risk", "experiment", "controls", "readout", "discriminative_value", "cost"):
            if e.get(k):
                parts.append(str(e[k]))
    for k in ("go_no_go", "sample_size_note"):
        if plan.get(k):
            parts.append(str(plan[k]))
    return "\n".join(parts)


def _exp_tokens(text: str) -> set[str]:
    """实验文本词元：英文词（≥2 字符）+ 中文双字滑动窗口（位置无关）。"""
    toks = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        toks.update(run[i:i + 2] for i in range(len(run) - 1))
    return {t for t in toks if t not in STOP}


def check_experiment_claim(text: str, experiment_plan: dict[str, Any] | None) -> tuple[bool, str]:
    """核对主张提到的实验名/判据是否出自本轮 design_experiment 输出。"""
    if experiment_plan is None:
        if any(m in text for m in _EXP_MARKERS):
            return False, "主张提到具体实验/判据，但未传入本轮实验设计输出，无法核对"
        return True, ""
    plan = _plan_text(experiment_plan)
    if not plan:
        return True, ""
    if any(m in text and m in plan for m in _EXP_MARKERS):
        return True, ""
    ctoks, ptoks = _exp_tokens(text), _exp_tokens(plan)
    if not ctoks:
        return True, ""
    if len(ctoks & ptoks) / len(ctoks) >= 0.25:
        return True, ""
    return False, "实验名/判据未见于本轮设计输出"


def verify_tool_claim(
    text: str,
    tool_cites: Iterable[str],
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> tuple[bool, float, str]:
    """校验含 [TOOL:...] 的主张：只凭“工具跑过”不够，数值与实验名必须与本轮工具输出对得上。

    返回 (grounded, overlap, reason)。无可核对内容时（如只有“通过”二字）返回 grounded。
    """
    cites = list(tool_cites)
    if "sequence_check" in cites:
        ok, reason = check_sequence_numbers(text, seq_metrics)
        if not ok:
            return False, 0.0, reason
    if "design_experiment" in cites:
        ok, reason = check_experiment_claim(text, experiment_plan)
        if not ok:
            return False, 0.0, reason
    return True, 1.0, ""

def _tokens(text: str) -> set[str]:
    """词元切分（S2）：英文词总长 ≥2（保留 ph/kd/vhh/egfr/aa 等短词）；
    中文用字符级 bigram 滑动窗口，替代位置相关的固定 2–4 字切块，
    使“提高表达量”与“表达量提高”能对上（交集非空）。"""
    en = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text.lower())
    zh: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        zh.update(run[i:i + 2] for i in range(len(run) - 1))
    return {t for t in en + list(zh) if t not in STOP}


_CJK_TOKEN_RE = re.compile(r"^[\u4e00-\u9fff]{2}$")


def _matchable_denominator(claim_toks: set[str], evidence_toks: set[str]) -> set[str]:
    """重叠分母：只计“可能匹配”的主张词元——英文词与术语桥展开（ASCII）全计；
    纯中文双字仅当其出现在证据词元中时才计（证据是英文摘要时，中文双字永远
    匹配不上，计入分母只会把 S02/S07/S08/S10 这类纯中文主张的分母虚增、
    把 direct 压成 indirect）。分母为空时返回空集，调用方按 0 分处理。"""
    return {t for t in claim_toks if not _CJK_TOKEN_RE.match(t) or t in evidence_toks}


def _overlap(claim_toks: set[str], evidence_toks: set[str]) -> tuple[float, int]:
    """返回（重叠率，分母词元数）。分母为空时重叠率为 0。"""
    den = _matchable_denominator(claim_toks, evidence_toks)
    if not den:
        return 0.0, 0
    return round(len(claim_toks & evidence_toks) / len(den), 3), len(den)


def _is_factual(line: str) -> bool:
    if len(line) < 12:
        return False
    if re.search(r"\d", line):
        return True
    return any(h in line for h in FACT_HINTS)


def extract_claims(note: str) -> list[Claim]:
    claims: list[Claim] = []
    for raw in note.splitlines():
        line = raw.strip()
        if not line or line.startswith(SKIP_PREFIX):
            continue
        body = re.sub(r"^[-*\d.、\s]+", "", line)
        if len(body) < 12:
            continue
        cites = ID_RE.findall(" ".join(CITE_RE.findall(body)))
        tool_cites = TOOL_CITE_RE.findall(body)
        # P2-1：图表引用 [P007:Fig1] 单独抽取（CITE_RE 不匹配 ":FigN" 后缀）
        fig_cites = [(pid, int(n)) for pid, n in FIG_CITE_RE.findall(body)]
        claims.append(
            Claim(text=body, cites=cites, tool_cites=tool_cites, fig_cites=fig_cites,
                  needs_evidence=_is_factual(body))
        )
    return claims


def ground_claims(
    note: str,
    evidence: Iterable[dict[str, Any]],
    tool_names: Iterable[str] = (),
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对每条主张做引用匹配与词级重叠校验。

    两类合法证据：
    - 文献证据 `[Pxxx]`：与该文献标题+摘要+关键发现做词级重叠（span matching），
      阈值按主张词元数自适应，且主张与证据方向严格相反时判未接地（M3）；
    - 工具证据 `[TOOL:sequence_check]` / `[TOOL:design_experiment]`：本轮确实执行过该工具
      只是必要条件，主张里的数值与实验名还须与本轮工具输出逐项核对（M2）。
    """
    tools = set(tool_names)
    ev = {str(e.get("id")): e for e in evidence if e.get("id")}
    ev_tokens = {}
    ev_blobs = {}
    for k, v in ev.items():
        blob = f"{v.get('title','')} {v.get('abstract','')} {' '.join(v.get('key_findings', []))}"
        ev_tokens[k] = _bridge(blob, _tokens(blob))
        ev_blobs[k] = blob

    claims = extract_claims(note)
    need = [c for c in claims if c.needs_evidence]

    for c in need:
        if c.tool_cites:
            unknown_tools = [t for t in c.tool_cites if t not in tools]
            if unknown_tools:
                c.reason = f"引用了本轮未执行的工具：{'、'.join(unknown_tools)}"
            else:
                ok, ov, why = verify_tool_claim(c.text, c.tool_cites, seq_metrics, experiment_plan)
                if not ok:
                    c.reason = why
                    c.overlap = 0.0
                else:
                    c.grounded = True
                    c.matched_id = "TOOL:" + c.tool_cites[0]
                    c.overlap = 1.0
                    c.reason = "工具徽章与本轮工具输出核对一致"
            continue
        # P2-1：图表引用 [Pxxx:FigN]——证据文本是该图的图注（caption），与文献摘要
        # 同一套词面重叠 + 自适应阈值 + 方向检查；图注取不到时判未接地，不回落到摘要。
        if c.fig_cites:
            pid, fno = c.fig_cites[0]
            caption = figure_caption(evidence, pid, fno)
            if caption is None:
                c.reason = f"引用图表 [{pid}:Fig{fno}]，但本轮证据中无该图表（论文未命中或无图表元数据）"
                continue
            ctoks = _bridge(c.text, _tokens(c.text))
            etoks = _bridge(caption, _tokens(caption))
            ov, n_den = _overlap(ctoks, etoks) if ctoks and etoks else (0.0, 0)
            threshold = overlap_threshold(n_den)
            conflict, why = directions_conflict(c.text, caption)
            if conflict:
                c.reason = f"引用 [{pid}:Fig{fno}]：{why}"
                c.overlap = round(ov, 3)
            elif ov < threshold:
                c.reason = f"引用 [{pid}:Fig{fno}] 图注与主张词面重叠仅 {ov:.0%}，低于 {threshold:.0%} 阈值"
                c.overlap = round(ov, 3)
            else:
                c.grounded = True
                c.matched_id = f"{pid}:Fig{fno}"
                c.overlap = round(ov, 3)
                c.reason = "引用图表图注与主张词面重叠且方向一致"
            continue
        if not c.cites:
            c.reason = "缺少引用标记 [Pxxx] 或 [TOOL:xxx]"
            continue
        unknown = [i for i in c.cites if i not in ev]
        if unknown:
            c.reason = f"引用了本轮未命中的 ID：{'、'.join(unknown)}"
            continue
        ctoks = _bridge(c.text, _tokens(c.text))
        # 按重叠度降序尝试各引用 ID：第一篇同时满足阈值与方向一致的即接地。
        # 阈值按“可匹配分母”（_overlap 的分母词元数）分档，而非全部词元数，
        # 否则纯中文主张的分母会被永远匹配不上的中文双字虚增。
        # 循环外的 threshold 是默认值（ASCII 词元数分档），供循环未执行时的理由文案使用。
        threshold = overlap_threshold(len([t for t in ctoks if not _CJK_TOKEN_RE.match(t)]))
        ranked = sorted(
            c.cites,
            key=lambda cid: len(ctoks & ev_tokens.get(cid, set())),
            reverse=True,
        )
        best_id, best = "", 0.0
        conflict_reason = ""
        for cid in ranked:
            toks = ev_tokens.get(cid, set())
            if not ctoks or not toks:
                continue
            ov, n_den = _overlap(ctoks, toks)
            threshold = overlap_threshold(n_den)
            if ov > best:
                best_id, best = cid, ov
            if ov < threshold:
                continue
            conflict, why = directions_conflict(c.text, ev_blobs.get(cid, ""))
            if conflict:
                conflict_reason = f"引用 {cid}：{why}"
                continue
            c.grounded = True
            c.matched_id = cid
            c.overlap = round(ov, 3)
            c.reason = "引用与主张词面重叠且方向一致"
            break
        if not c.grounded:
            c.overlap, c.matched_id = round(best, 3), best_id
            if conflict_reason:
                c.reason = conflict_reason
            else:
                ref = best_id or c.cites[0]
                c.reason = f"引用 {ref} 与主张词面重叠仅 {best:.0%}，低于 {threshold:.0%} 阈值，疑似张冠李戴"

    grounded = sum(1 for c in need if c.grounded)
    score = round(100 * grounded / len(need)) if need else 100
    return {
        "claims_total": len(claims),
        "claims_need_evidence": len(need),
        "claims_grounded": grounded,
        "grounding_score": score,
        "ungrounded": [
            {"text": c.text[:160], "cites": c.cites, "reason": c.reason, "overlap": c.overlap}
            for c in need if not c.grounded
        ][:12],
        "evidence_ids": sorted(ev),
    }


GROUND_THRESHOLD = 0.24  # reground 重匹配阈值（比接地判定更严，避免把弱匹配"洗白"成引用）


def _evidence_tokens(evidence: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in evidence:
        if not e.get("id"):
            continue
        blob = f"{e.get('title','')} {e.get('abstract','')} {' '.join(e.get('key_findings', []))}"
        out[str(e.get("id"))] = _bridge(blob, _tokens(blob))
    return out


def best_snippet(text: str, ev_tokens: dict[str, set[str]]) -> tuple[str, float]:
    """按词级重叠找最匹配的检索片段；平局按 ID 字典序，保证确定性。"""
    clean = re.sub(r"\[[^\]]*\]", "", text)
    ctoks = _bridge(clean, _tokens(clean))
    best_id, best = "", 0.0
    for cid in sorted(ev_tokens):
        toks = ev_tokens[cid]
        if not ctoks or not toks:
            continue
        ov, _ = _overlap(ctoks, toks)
        if ov > best:
            best_id, best = cid, ov
    return best_id, best


def reground_note(
    note: str,
    evidence: Iterable[dict[str, Any]],
    tool_names: Iterable[str] = (),
    threshold: float = GROUND_THRESHOLD,
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """确定性引用后处理（不再依赖模型"听话"）。

    对每条需要证据却未接地的事实性主张行：
    - 若与本轮某个检索片段的词面重叠 ≥ 阈值（默认 0.24） → 把引用替换为该片段 ID（re-match）；
    - 否则 → 删除该行并记入 dropped（flag & remove）。
    已接地的行、工具引用合法的行、非事实性行原样保留。行内的 [TOOL:...] 徽章不再 strip，
    以免把工具数值核对失败的行伪装成普通文献主张。返回 (新笔记, 统计)。
    """
    ev = list(evidence)
    tools = list(tool_names)
    g_before = ground_claims(note, ev, tools, seq_metrics, experiment_plan) if ev else None
    before = g_before["grounding_score"] if g_before else 0
    ev_tok = _evidence_tokens(ev)
    stats: dict[str, Any] = {
        "grounding_before": before,
        "claims_need_before": g_before["claims_need_evidence"] if g_before else 0,
        "claims_grounded_before": g_before["claims_grounded"] if g_before else 0,
        "rematched": [],
        "dropped": [],
    }
    if not ev_tok:
        stats["grounding_after"] = before
        stats["claims_need_after"] = stats["claims_need_before"]
        stats["claims_grounded_after"] = stats["claims_grounded_before"]
        stats["dropped_count"] = 0
        stats["rematched_count"] = 0
        return note, stats
    out: list[str] = []
    for raw in note.splitlines():
        line = raw.strip()
        if not line or line.startswith(SKIP_PREFIX):
            out.append(raw); continue
        claims = extract_claims(raw)
        if not claims or not claims[0].needs_evidence:
            out.append(raw); continue
        g = ground_claims(raw, ev, tools, seq_metrics, experiment_plan)
        if g["claims_need_evidence"] == 0 or g["claims_grounded"] == g["claims_need_evidence"]:
            out.append(raw); continue
        # P2-1：图表主张 [Pxxx:FigN] 未接地时直接删除，不重匹配为文献引用
        # （图注证据与文献摘要证据类型不同，不可混用；strip 正则也匹配不到 Fig 引用）
        if claims[0].fig_cites:
            stats["dropped"].append({"text": line[:80], "best": None, "overlap": 0.0,
                                     "reason": "图表主张未接地，已删除（不重匹配为文献引用）"})
            continue
        cid, ov = best_snippet(claims[0].text, ev_tok)
        if cid and ov >= threshold:
            stripped = re.sub(r"\s*\[(?:P\d{3}(?:\s*[,;，、]\s*P\d{3})*)\]", "", raw).rstrip()
            m = re.search(r"[。．.！!？?；;]$", stripped)
            new = (stripped[:-1] + f" [{cid}]" + stripped[-1]) if m else f"{stripped} [{cid}]"
            out.append(new)
            stats["rematched"].append({"text": line[:80], "to": cid, "overlap": round(ov, 3)})
        else:
            stats["dropped"].append({"text": line[:80], "best": cid, "overlap": round(ov, 3)})
    new_note = "\n".join(out) + ("\n" if note.endswith("\n") else "")
    g_after = ground_claims(new_note, ev, tools, seq_metrics, experiment_plan)
    stats["grounding_after"] = g_after["grounding_score"]
    stats["claims_need_after"] = g_after["claims_need_evidence"]
    stats["claims_grounded_after"] = g_after["claims_grounded"]
    stats["dropped_count"] = len(stats["dropped"])
    stats["rematched_count"] = len(stats["rematched"])
    return new_note, stats
