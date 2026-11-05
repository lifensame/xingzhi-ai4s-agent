"""证据辩证层：引用支撑强度分级 + 过度表述降级 + 文献分歧检测。

设计原则与 grounding.py 一致：全部确定性、纯标准库，离线可复现。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .grounding import (
    _tokens, _bridge, _overlap, extract_claims,
    direction_of, directions_conflict, overlap_threshold, verify_tool_claim,
    _SEQ_NUM_PATTERNS, _CHARGE_RE, _EXP_MARKERS,
    UP_ZH, DOWN_ZH, UP_EN, DOWN_EN,
)

# 强断言词 → 降级后的稳健表述
SOFTEN = {
    "证明": "提示",
    "必然": "可能",
    "完全": "在本文条件下基本",
    "一定会": "可能会",
    "总是": "在多数报道中",
    "彻底": "较大程度地",
    "proves": "is consistent with",
    "causes": "is associated with",
    "always": "in most reports",
    "guarantees": "suggests",
}
STRONG = tuple(SOFTEN)
# 支撑强度阈值
DIRECT_T = 0.24
# 模糊限定词：命中时即便词面重叠高，也只记作间接支撑
HEDGE = ("可以", "有助", "辅助", "或许", "倾向", "大致", "一定程度")
INDIRECT_T = 0.12

LEVELS = ("direct", "indirect", "overstated", "unsupported")
LEVEL_CN = {
    "direct": "直接支持",
    "indirect": "间接支持",
    "overstated": "过度表述",
    "unsupported": "无支撑",
}





@dataclass
class SupportRow:
    text: str
    level: str
    overlap: float
    matched_id: str = ""
    softened: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:200],
            "level": self.level,
            "level_cn": LEVEL_CN[self.level],
            "overlap": self.overlap,
            "matched_id": self.matched_id,
            "softened": self.softened,
            "reason": self.reason,
        }


# 否定标记：强断言词前约 6 字内出现这些词时，该强断言词不判过度表述、不改写
_NEG_BEFORE_ZH = ("不", "未", "没", "非", "否", "并不", "并非", "不是", "不会", "没有", "很少")
_NEG_BEFORE_EN = ("not ", "never ", "n't", "rarely ", "no ")


def _is_negated(text: str, idx: int) -> bool:
    before = text[max(0, idx - 6):idx]
    return any(n in before for n in _NEG_BEFORE_ZH) or any(n in before.lower() for n in _NEG_BEFORE_EN)


def _find_occurrences(text: str, word: str) -> list[int]:
    """返回 word 在 text 中的出现位置；英文按词边界匹配，避免 improves 误中 proves。"""
    is_en = bool(re.fullmatch(r"[A-Za-z]+", word))
    out: list[int] = []
    start = 0
    while True:
        i = text.find(word, start)
        if i < 0:
            return out
        if is_en:
            before_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            after_ok = i + len(word) >= len(text) or not (text[i + len(word)].isalnum() or text[i + len(word)] == "_")
            if before_ok and after_ok:
                out.append(i)
        else:
            out.append(i)
        start = i + len(word)


def strong_words(text: str) -> list[str]:
    """否定句中的强断言词不计入（如"结果并不完全一致"里的"完全"）。"""
    return [w for w in STRONG if any(not _is_negated(text, i) for i in _find_occurrences(text, w))]


def _replace_strong(text: str, word: str, repl: str) -> str:
    """只替换未被否定的出现；"证明文件"等词组不改写（证明后接 文件/材料/书/信 时跳过）。"""
    blocked_after = {"证明": "文件材料书信", "完全": "性"}.get(word, "")
    parts, cur = [], 0
    for i in _find_occurrences(text, word):
        if blocked_after and i + len(word) < len(text) and text[i + len(word)] in blocked_after:
            continue
        if _is_negated(text, i):
            continue
        parts.append(text[cur:i])
        parts.append(repl)
        cur = i + len(word)
    parts.append(text[cur:])
    return "".join(parts)


def soften(text: str) -> str:
    """降级强断言词；英文按词边界替换，避免误伤 completely/always 等子串。"""
    out = text
    for strong, mild in SOFTEN.items():
        out = _replace_strong(out, strong, mild)
    return out


def _has_sequence_number(text: str) -> bool:
    return any(p.search(text) for p, _, _ in _SEQ_NUM_PATTERNS) or bool(_CHARGE_RE.search(text))


def _has_experiment_marker(text: str) -> bool:
    return any(m in text for m in _EXP_MARKERS)


def grade_support(sentence: str, snippet: str) -> tuple[str, float, str]:
    """返回 (等级, 重叠度, 理由)。确定性，不调用模型。"""
    clean = re.sub(r"\[[^\]]*\]", "", sentence)
    stoks = _bridge(clean, _tokens(clean))
    etoks = _bridge(snippet, _tokens(snippet))
    # 重叠率与阈值分档都用“可匹配分母”（见 grounding._overlap），与 ground_claims 同口径
    ov, n_den = _overlap(stoks, etoks) if stoks and etoks else (0.0, 0)
    indirect_t = overlap_threshold(n_den) if n_den else INDIRECT_T
    direct_t = max(DIRECT_T, indirect_t)
    if ov < indirect_t:
        return "unsupported", ov, f"与片段词面重叠仅 {ov:.0%}，低于 {indirect_t:.0%} 阈值"
    conflict, why = directions_conflict(sentence, snippet)
    if conflict:
        return "unsupported", ov, why
    sw = strong_words(sentence)
    if sw and not any(w in snippet for w in sw):
        return (
            "overstated",
            ov,
            f"句中使用强断言词「{ '、'.join(sw) }」，但引用片段未给出同等强度的表述",
        )
    if ov >= direct_t:
        hedge = [h for h in HEDGE if h in sentence]
        if hedge:
            return "indirect", ov, f"重叠 {ov:.0%} 但句中含模糊限定词「{'、'.join(hedge)}」，记作间接支撑"
        return "direct", ov, f"与片段词面重叠 {ov:.0%}，要点可逐词对上"
    return "indirect", ov, f"与片段相关但重叠仅 {ov:.0%}，属背景性支撑"


def grade_note(
    note: str,
    evidence: Iterable[dict[str, Any]],
    tool_names: Iterable[str] | None = None,
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """逐句分级，并把过度表述的句子改写为稳健措辞。返回统计与新笔记。

    口径与 grounding.py 一致：无引用主张直接记 unsupported，不再全库回退找分；
    多引用时取支撑最强的 ID；含 [TOOL:...] 的主张须与本轮工具输出逐项核对，
    未传入工具输出时降级为间接支撑（不直接记 direct，也不硬判无支撑）。
    """
    ev = {str(e.get("id")): e for e in evidence if e.get("id")}
    tools = {t.lower() for t in tool_names} if tool_names is not None else None
    rows: list[SupportRow] = []
    new_note = note
    for c in extract_claims(note):
        if not c.needs_evidence:
            continue
        if c.tool_cites:
            if tools is not None:
                unknown = [t for t in c.tool_cites if t.lower() not in tools]
                if unknown:
                    rows.append(SupportRow(
                        text=c.text, level="unsupported", overlap=0.0, matched_id="",
                        reason=f"引用了本轮未执行的工具 {unknown}",
                    ))
                    continue
            if "sequence_check" in c.tool_cites and seq_metrics is None and _has_sequence_number(c.text):
                rows.append(SupportRow(
                    text=c.text, level="indirect", overlap=0.5, matched_id="TOOL:sequence_check",
                    reason="工具已执行，但未传入本轮理化计算结果，数值未逐项核对，记间接支撑",
                ))
                continue
            if "design_experiment" in c.tool_cites and experiment_plan is None and _has_experiment_marker(c.text):
                rows.append(SupportRow(
                    text=c.text, level="indirect", overlap=0.5, matched_id="TOOL:design_experiment",
                    reason="工具已执行，但未传入本轮实验设计输出，实验名/判据未逐项核对，记间接支撑",
                ))
                continue
            ok, _, why = verify_tool_claim(c.text, c.tool_cites, seq_metrics, experiment_plan)
            if not ok:
                rows.append(SupportRow(
                    text=c.text, level="unsupported", overlap=0.0,
                    matched_id="TOOL:" + c.tool_cites[0], reason=why,
                ))
            else:
                rows.append(SupportRow(
                    text=c.text, level="direct", overlap=1.0,
                    matched_id="TOOL:" + c.tool_cites[0],
                    reason="由本轮实际执行的计算工具给出，数值可重算核对",
                ))
            continue
        # P2-1：图表引用 [Pxxx:FigN]——按图注文本分级，与 ground_claims 同口径；
        # 图注取不到时记 unsupported，不回落到文献摘要。
        if c.fig_cites:
            from .figures import figure_caption as _fig_caption
            pid, fno = c.fig_cites[0]
            caption = _fig_caption(evidence, pid, fno)
            if caption is None:
                rows.append(SupportRow(
                    text=c.text, level="unsupported", overlap=0.0,
                    matched_id=f"{pid}:Fig{fno}",
                    reason=f"引用图表 [{pid}:Fig{fno}]，但本轮证据中无该图表（论文未命中或无图表元数据）",
                ))
            else:
                level, ov, reason = grade_support(c.text, caption)
                rows.append(SupportRow(
                    text=c.text, level=level, overlap=ov,
                    matched_id=f"{pid}:Fig{fno}", reason=reason,
                ))
            continue
        cited = [i for i in c.cites if i in ev]
        if not cited:
            rows.append(SupportRow(
                text=c.text, level="unsupported", overlap=0.0,
                matched_id=c.cites[0] if c.cites else "",
                reason="主张无任何文献引用" if not c.cites else f"引用的文献 ID {c.cites} 不在本轮证据中",
            ))
            continue
        # 多引用时取支撑最强的 ID（direct > indirect > overstated > unsupported，同级取重叠最高）
        rank = {"direct": 3, "indirect": 2, "overstated": 1, "unsupported": 0}
        best_row: SupportRow | None = None
        best_rank = -1
        for cid in cited:
            src = ev[cid]
            snippet = f"{src.get('title','')} {src.get('abstract','')} {' '.join(src.get('key_findings', []))}"
            level, ov, reason = grade_support(c.text, snippet)
            if rank[level] > best_rank or (rank[level] == best_rank and best_row is not None and ov > best_row.overlap):
                best_rank = rank[level]
                best_row = SupportRow(text=c.text, level=level, overlap=ov, matched_id=cid, reason=reason)
        row = best_row or SupportRow(text=c.text, level="unsupported", overlap=0.0, reason="无可核对引用")
        if row.level == "overstated":
            row.softened = soften(c.text)
            new_note = new_note.replace(c.text, row.softened)
        rows.append(row)

    counts = {lv: sum(1 for r in rows if r.level == lv) for lv in LEVELS}
    total = len(rows) or 1
    return {
        "rows": [r.as_dict() for r in rows],
        "counts": counts,
        "softened": [{"before": r.text[:160], "after": r.softened[:160]} for r in rows if r.softened],
        "support_score": round(100 * (counts["direct"] + 0.5 * counts["indirect"]) / total),
        "note_markdown": new_note,
    }


# ---------- 文献分歧检测 ----------

# 方向词表已收敛到 grounding.py（direction_of），此处不再另维护一份。
METHOD_WORDS = ("计算", "预测", "模拟", "实验", "筛选", "测定", "综述", "机器学习", "结构",
                "in silico", "in vitro", "in vivo", "review", "assay", "simulation", "model")
COND_RE = re.compile(r"(pH\s?\d+(?:\.\d+)?|\d+\s?(?:°C|℃|mM|nM|µM|uM|kDa|aa|bp))")
# 注意：COND_RE 不含年份——发表年份不是实验条件，两个同年发表的结论不应因此被判"条件相同"。


@dataclass
class Statement:
    pid: str
    topic: str
    direction: str
    conditions: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid, "topic": self.topic, "direction": self.direction,
            "conditions": self.conditions, "methods": self.methods, "text": self.text[:180],
        }


def _direction(text: str) -> str:
    """方向判定与 grounding.py 共用同一口径；双向表述返回 mixed，参与分歧配对。"""
    return direction_of(text)


def _statement_direction(text: str) -> str:
    """单篇文献的陈述方向（供 extract_statements 使用）。

    _direction() 会把双向表述记为 mixed，但下游的假设锦标赛需要单一方向做红队计票
    （mixed 既不等于 up 也不等于 down，会导致主张方文献被误记入反证，且同一文献可能
    同时出现在支持与反证两侧）。因此这里对 mixed 按方向词出现次数多数决，
    平局取文本中首次出现的方向词方向，保证返回 up/down/neutral。
    原始双向性仍由 _direction() 保留，classify_pair 对手工构造的 mixed 陈述依然有效。
    """
    d = _direction(text)
    if d != "mixed":
        return d
    low = text.lower()
    en_toks = re.findall(r"[a-z][a-z\-]{2,}", low)
    ups = sum(text.count(w) for w in UP_ZH) + sum(1 for t in en_toks if t in UP_EN)
    downs = sum(text.count(w) for w in DOWN_ZH) + sum(1 for t in en_toks if t in DOWN_EN)
    if ups > downs:
        return "up"
    if downs > ups:
        return "down"
    cands: list[tuple[int, str]] = []
    for w in UP_ZH:
        i = text.find(w)
        if i >= 0:
            cands.append((i, "up"))
    for w in DOWN_ZH:
        i = text.find(w)
        if i >= 0:
            cands.append((i, "down"))
    for m in re.finditer(r"[a-z][a-z\-]{2,}", low):
        if m.group() in UP_EN:
            cands.append((m.start(), "up"))
        elif m.group() in DOWN_EN:
            cands.append((m.start(), "down"))
    return min(cands)[1] if cands else "neutral"


def extract_statements(papers: Iterable[dict[str, Any]]) -> list[Statement]:
    out: list[Statement] = []
    for p in papers:
        text = " ".join(p.get("key_findings") or []) + " " + (p.get("abstract") or "")
        text = text.strip()
        if not text:
            continue
        low = text.lower()
        out.append(
            Statement(
                pid=str(p.get("id") or p.get("pmid") or p.get("doi") or ""),
                topic=str(p.get("topic") or "未分类"),
                direction=_statement_direction(text),
                conditions=sorted(set(COND_RE.findall(text)))[:4],
                methods=sorted({m for m in METHOD_WORDS if m in low or m in text})[:4],
                text=text,
            )
        )
    return out


def classify_pair(a: Statement, b: Statement) -> tuple[str, str] | None:
    """同主题且方向相反时，判定分歧类型。返回 (类型, 理由) 或 None。

    mixed（含双向表述）与纯 up / 纯 down 均可构成冲突可能；neutral 仍排除。
    """
    if a.topic != b.topic or "neutral" in (a.direction, b.direction):
        return None
    if a.direction == b.direction:
        return None
    ca, cb = set(a.conditions), set(b.conditions)
    if ca and cb and not (ca & cb):
        return "conditions", f"实验条件不同（{'、'.join(sorted(ca))} vs {'、'.join(sorted(cb))}），属条件差异导致的表观冲突"
    ma, mb = set(a.methods), set(b.methods)
    if ma and mb and not (ma & mb):
        return "definitions", f"口径/方法不同（{'、'.join(sorted(ma))} vs {'、'.join(sorted(mb))}），指标定义可能不可直接比较"
    return "real", "条件与方法口径一致，但结论方向相反，属实质性分歧"


DISAGREE_CN = {
    "real": "实质分歧",
    "conditions": "条件差异（伪冲突）",
    "definitions": "口径差异（伪冲突）",
}


def detect_disagreements(papers: Iterable[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    stmts = extract_statements(papers)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(stmts):
        for b in stmts[i + 1:]:
            hit = classify_pair(a, b)
            if hit:
                kind, reason = hit
                pairs.append({
                    "kind": kind, "kind_cn": DISAGREE_CN[kind], "reason": reason,
                    "topic": a.topic, "a": a.pid, "b": b.pid,
                })
    pairs.sort(key=lambda d: (d["kind"] != "real", d["topic"], d["a"], d["b"]))
    counts = {k: sum(1 for p in pairs if p["kind"] == k) for k in DISAGREE_CN}
    return {
        "statements": [s.as_dict() for s in stmts],
        "disagreements": pairs[:limit],
        "disagreements_all": pairs,
        "counts": counts,
        "topics": sorted({s.topic for s in stmts}),
    }


def confidence(topic: str, papers: Iterable[dict[str, Any]], disagreements: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """按来源数量、方法多样性、分歧情况给出 0–100 置信度与等级。"""
    stmts = [s for s in extract_statements(papers) if s.topic == topic]
    n = len(stmts)
    methods = {m for s in stmts for m in s.methods}
    real = sum(1 for d in disagreements if d["topic"] == topic and d["kind"] == "real")
    # O5：基线从 0 起按证据量累加（去掉原来的 +20），零文献主题为 0 分，
    # 单篇无方法词主题为 10 分；等级阈值不变（高 ≥70，中 ≥40）。
    score = min(50, 10 * n) + min(30, 10 * len(methods)) - min(40, 20 * real)
    score = max(0, min(100, score))
    level = "高" if score >= 70 else ("中" if score >= 40 else "低")
    return {
        "topic": topic, "confidence": score, "level": level,
        "sources": n, "method_variety": len(methods), "real_conflicts": real,
    }


def analyze(
    note: str,
    evidence: Iterable[dict[str, Any]],
    corpus: Iterable[dict[str, Any]] | None = None,
    tool_names: Iterable[str] | None = None,
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """支撑分级只看本轮检索到的证据；分歧检测在整个语料上做，覆盖面更大。"""
    ev = list(evidence)
    support = grade_note(note, ev, tool_names=tool_names, seq_metrics=seq_metrics, experiment_plan=experiment_plan)
    dis = detect_disagreements(list(corpus) if corpus is not None else ev)
    pool = list(corpus) if corpus is not None else ev
    confs = [confidence(t, pool, dis.get("disagreements_all", dis["disagreements"])) for t in dis["topics"]]
    confs.sort(key=lambda d: (-d["confidence"], d["topic"]))
    return {
        "support": {k: v for k, v in support.items() if k != "note_markdown"},
        "note_markdown": support["note_markdown"],
        "disagreements": dis,
        "confidence": confs,
    }
