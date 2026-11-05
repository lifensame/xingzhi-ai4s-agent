"""假设锦标赛：从文献分歧/缺口生成可检验假设 → 红队找反证 → 两两对决 Elo 排名。

离线模式用确定性规则，LIVE 模式可选用模型改写措辞；排序与打分始终由确定性规则决定，
保证"同输入 → 同排名"，可复现、可审计。
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .evidence import DISAGREE_CN, Statement, _bridge, detect_disagreements, extract_statements
from .grounding import _tokens

ELO_START = 1000.0
ELO_K = 24.0


@dataclass
class Hypothesis:
    hid: str
    text: str
    origin: str                      # conflict / gap
    topic: str
    claim_direction: str = "neutral"  # 假设主张的方向：up / down / neutral
    claim_pids: list[str] = field(default_factory=list)  # 主张成立的一方文献
    oppose_pids: list[str] = field(default_factory=list)  # 对立一方文献
    support_ids: list[str] = field(default_factory=list)
    counter_ids: list[str] = field(default_factory=list)
    elo: float = ELO_START
    wins: int = 0
    losses: int = 0

    @property
    def survival(self) -> int:
        """存活分：支持与反证之比，叠加 Elo 相对强度，截断到 0–100。"""
        s, c = len(self.support_ids), len(self.counter_ids)
        base = 100 * (s + 1) / (s + 2 * c + 2)
        return max(0, min(100, round(base + (self.elo - ELO_START) / 4)))

    @property
    def contest(self) -> int:
        """双方都有证据时的对抗强度，用于判定"最需要做判别实验"的主题。"""
        return min(len(self.support_ids), len(self.counter_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.hid, "text": self.text, "origin": self.origin, "topic": self.topic,
            "claim_direction": self.claim_direction,
            "claim": self.claim_pids, "oppose": self.oppose_pids,
            "support": self.support_ids, "counter": self.counter_ids,
            "elo": round(self.elo, 1), "wins": self.wins, "losses": self.losses,
            "survival": self.survival, "contest": self.contest,
        }


def _topic_words(topic: str) -> set[str]:
    return {w for w in re.split(r"[_\s\-]+", topic.lower()) if len(w) > 2}


def _dir_of(pid: str, stmts: list[Statement]) -> str:
    for s in stmts:
        if s.pid == pid:
            return s.direction
    return "neutral"


def _modal_direction(stmts: list[Statement]) -> str:
    counts = Counter(s.direction for s in stmts if s.direction in ("up", "down"))
    if not counts:
        return "neutral"
    # 票多者；平局取 up（确定性）
    best = max(counts.values())
    cands = sorted(d for d, n in counts.items() if n == best)
    return cands[0] if len(cands) == 1 else ("up" if "up" in cands else cands[0])


def _gap_method_phrase(stmts: list[Statement]) -> str:
    methods = sorted({m for s in stmts for m in s.methods})
    n = len(methods)
    if n <= 1:
        label = methods[0] if methods else "未标注"
        return f"且方法单一（仅见「{label}」）"
    if n == 2:
        return f"且仅覆盖 {n} 类方法（{'、'.join(methods)}），仍缺独立复现"
    return f"虽覆盖 {n} 类方法（{'、'.join(methods[:3])}…）但同向独立证据仍偏少"


def generate(corpus: Iterable[dict[str, Any]], limit: int = 6) -> list[Hypothesis]:
    """来源一：文献分歧 → 「谁对」的假设；来源二：证据稀薄的主题 → 「缺口」假设。

    同一 (topic, claim_direction) 只生成一条分歧型假设，避免重复立场占满名额。
    """
    papers = list(corpus)
    dis = detect_disagreements(papers, limit=50)
    stmts = extract_statements(papers)
    by_topic: dict[str, list[Statement]] = {}
    for s in stmts:
        by_topic.setdefault(s.topic, []).append(s)

    out: list[Hypothesis] = []
    seen_conflict: set[tuple[str, str]] = set()
    for d in dis["disagreements"]:
        if len(out) >= limit:
            break
        claim_dir = _dir_of(d["a"], stmts)
        if claim_dir == "neutral":
            # 主张方无方向时，取对立方的反方向
            opp = _dir_of(d["b"], stmts)
            claim_dir = "down" if opp == "up" else ("up" if opp == "down" else "neutral")
        key = (d["topic"], claim_dir)
        if key in seen_conflict:
            continue
        seen_conflict.add(key)
        # 同主题同方向的对立文献一并收进 oppose
        oppose = [d["b"]]
        for other in dis["disagreements"]:
            if other["topic"] == d["topic"] and other["a"] == d["a"] and other["b"] not in oppose:
                oppose.append(other["b"])
        out.append(Hypothesis(
            hid=f"H{len(out)+1:02d}",
            text=(f"在「{d['topic']}」上，{d['a']} 的方向性结论成立，而 "
                  f"{'、'.join(oppose)} 的相反结论源于{DISAGREE_CN[d['kind']]}；"
                  f"可通过统一条件与口径后复测来判定。"),
            origin="conflict", topic=d["topic"],
            claim_direction=claim_dir,
            claim_pids=[d["a"]],
            oppose_pids=oppose,
        ))

    # 缺口：按证据条数升序，优先补最薄的主题；已出过分歧假设的主题跳过
    conflict_topics = {h.topic for h in out}
    for topic in sorted(by_topic, key=lambda t: (len(by_topic[t]), t)):
        if len(out) >= limit:
            break
        if topic in conflict_topics:
            continue
        bucket = by_topic[topic]
        n = len(bucket)
        claim_dir = _modal_direction(bucket)
        phrase = _gap_method_phrase(bucket)
        out.append(Hypothesis(
            hid=f"H{len(out)+1:02d}",
            text=(f"「{topic}」目前有 {n} 条证据支撑{phrase}，若补充独立方法的重复实验，"
                  f"现有方向性结论仍将成立。"),
            origin="gap", topic=topic,
            claim_direction=claim_dir,
            claim_pids=[s.pid for s in bucket if s.direction == claim_dir][:4],
        ))
    return out[:limit]


def red_team(h: Hypothesis, corpus: Iterable[dict[str, Any]]) -> Hypothesis:
    """红队：相对假设的 claim_direction 计票——同向为 support，反向为 counter。

    主张方 / 对立方文献即使词面弱重叠也强制入账；claim_direction 为中性时，
    有方向的同主题文献记入 counter（待核对），避免无主张方向时误标为支持。
    """
    htoks = _bridge(h.text, _tokens(h.text)) | _topic_words(h.topic)
    claim = h.claim_direction
    for s in extract_statements(corpus):
        if s.topic != h.topic:
            continue
        if s.direction == "neutral":
            continue
        stoks = _bridge(s.text, _tokens(s.text))
        # 冲突型：主张方/对立方文献即使词面弱重叠也强制入账
        forced = s.pid in h.claim_pids or s.pid in h.oppose_pids
        if not forced and not (htoks & stoks):
            continue
        if claim in ("up", "down"):
            if s.direction == claim:
                h.support_ids.append(s.pid)
            else:
                h.counter_ids.append(s.pid)
        else:
            # 中性主张（主题全中性）：有方向的文献一律记为"待核对的对立证据"
            h.counter_ids.append(s.pid)
    h.support_ids = sorted(set(h.support_ids))[:8]
    h.counter_ids = sorted(set(h.counter_ids))[:8]
    return h


def _default_judge(a: Hypothesis, b: Hypothesis) -> str:
    """确定性裁判：优先"最需要做判别实验"的假设。

    1) 双方都有证据的对抗强度（contest）高者胜
    2) 分歧型优先于缺口型
    3) 证据净值（支持 − 反证）高者胜
    4) ID 字典序
    """
    if a.contest != b.contest:
        return a.hid if a.contest > b.contest else b.hid
    if a.origin != b.origin:
        return a.hid if a.origin == "conflict" else b.hid
    ka = len(a.support_ids) - len(a.counter_ids)
    kb = len(b.support_ids) - len(b.counter_ids)
    if ka != kb:
        return a.hid if ka > kb else b.hid
    return min(a.hid, b.hid)


def tournament(
    hyps: list[Hypothesis],
    judge: Callable[[Hypothesis, Hypothesis], str] | None = None,
) -> list[dict[str, Any]]:
    """循环赛：每对比一次，按 Elo 更新。返回逐场记录。"""
    judge = judge or _default_judge
    by_id = {h.hid: h for h in hyps}
    log: list[dict[str, Any]] = []
    for a, b in itertools.combinations(sorted(hyps, key=lambda h: h.hid), 2):
        winner = judge(a, b)
        if winner not in (a.hid, b.hid):
            winner = _default_judge(a, b)
        w, l = (a, b) if winner == a.hid else (b, a)
        exp_w = 1 / (1 + 10 ** ((l.elo - w.elo) / 400))
        w.elo += ELO_K * (1 - exp_w)
        l.elo -= ELO_K * (1 - exp_w)
        w.wins += 1
        l.losses += 1
        log.append({"a": a.hid, "b": b.hid, "winner": w.hid,
                    "elo_a": round(by_id[a.hid].elo, 1), "elo_b": round(by_id[b.hid].elo, 1)})
    return log


def next_experiment(h: Hypothesis) -> dict[str, Any]:
    """给排名第一的假设一条最小判别性下一步。"""
    return {
        "hypothesis": h.hid,
        "design": (f"在统一条件与统一指标口径下，对「{h.topic}」做一次最小判别实验："
                   f"取支持组文献 {('、'.join(h.support_ids[:3]) or '—')} 与反证组 "
                   f"{('、'.join(h.counter_ids[:3]) or '—')} 各自报告的条件，设为两个对照臂，"
                   f"同一批次、同一测定方法下重测该主题的方向性指标。"),
        "readout": "两臂方向是否一致；若一致则原分歧归因于条件/口径差异，若相反则为实质分歧。",
        "decision": "方向一致 → 合并结论并提高置信度；方向相反 → 该主题结论降级为待定，不进入下游设计。",
    }


def run(corpus: Iterable[dict[str, Any]], limit: int = 6,
        judge: Callable[[Hypothesis, Hypothesis], str] | None = None) -> dict[str, Any]:
    papers = list(corpus)
    hyps = [red_team(h, papers) for h in generate(papers, limit=limit)]
    if not hyps:
        return {"hypotheses": [], "matches": [], "next_experiment": None}
    matches = tournament(hyps, judge=judge)
    # 排名：Elo 为主；并列时对抗强度高、分歧型优先
    ranked = sorted(hyps, key=lambda h: (-h.elo, -h.contest, 0 if h.origin == "conflict" else 1, h.hid))
    return {
        "hypotheses": [h.as_dict() for h in ranked],
        "matches": matches,
        "rounds": len(matches),
        "next_experiment": next_experiment(ranked[0]),
    }
