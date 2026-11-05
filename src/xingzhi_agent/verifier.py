"""校验器 / Verifier：规则检查 + 引用接地（claim-level grounding）+ 修复反馈。

总分 = 规则分 × 0.5 + 引用接地分 × 0.5
并产出 `repair_feedback`：给自修复环（critic-repair loop）的具体整改指令。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .grounding import ground_claims

BANNED = [
    "官方指定", "主办方认证", "组委会认证", "保证获奖", "必定获奖",
    "真实湿实验已测", "已通过临床", "官方合作",
]
REQUIRED_SECTIONS = ["意图", "计划", "证据", "校验", "风险"]
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{12,}|eyJ[A-Za-z0-9._\-]{20,})")
CITE_RE = re.compile(r"P\d{3}")
GROUNDING_TARGET = 0.85  # 期望至少 85% 的事实性主张被接地


def verify_note_claims(
    note_body: str,
    *,
    literature_ids: list[str] | None = None,
    sequence_ok: bool | None = None,
    evidence: Iterable[dict[str, Any]] | None = None,
    tool_names: Iterable[str] = (),
    physchem_ok: bool | None = None,
    seq_metrics: dict[str, Any] | None = None,
    experiment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            (issues if hard else warnings).append(f"{name}：{detail}" if detail else name)

    hit_banned = [b for b in BANNED if b in note_body]
    check("合规表述", not hit_banned, "含不当表述 " + "、".join(hit_banned) if hit_banned else "")
    check("安全扫描（无疑似密钥）", not SECRET_RE.search(note_body), "笔记中出现疑似密钥字符串")

    has_disclaimer = any(k in note_body for k in ("演示", "Demo", "声明"))
    check("演示用途声明", has_disclaimer, "建议标明演示 / 非湿实验用途", hard=False)

    missing = [s for s in REQUIRED_SECTIONS if s not in note_body]
    check("结构完整", not missing, "缺少小节 " + "、".join(missing) if missing else "", hard=False)

    cited = sorted(set(CITE_RE.findall(note_body)))
    if literature_ids:
        unknown = sorted(set(cited) - set(literature_ids))
        check("引用可追溯", not unknown,
              "引用了本轮未命中的 ID " + "、".join(unknown) if unknown else "", hard=False)
        check("证据覆盖", bool(cited), "笔记未引用任何检索命中 ID", hard=False)
    else:
        check("证据覆盖", False, "本轮未记录文献命中 ID", hard=False)

    if sequence_ok is False:
        # 诚实记录失败不算硬失败——与理化结论一致性同口径
        honest = any(k in note_body for k in ("序列校验未通过", "序列未通过", "序列校验失败", "未通过序列"))
        check("序列校验一致性", honest,
              "" if honest else "序列校验未通过，但笔记未如实记录失败",
              hard=False)
    elif sequence_ok is True:
        check("序列校验一致性", True)

    if physchem_ok is not None:
        check("理化结论一致性", bool(physchem_ok) or "未通过" in note_body,
              "理化校验未通过，但笔记未如实记录", hard=False)

    check("篇幅充分", len(note_body) >= 900, f"笔记仅 {len(note_body)} 字符，内容偏薄", hard=False)
    check("含下一步实验设计", any(k in note_body for k in ("下一步实验", "实验设计", "Go / No-Go")),
          "缺少可执行的下一步实验建议", hard=False)

    ev = list(evidence or [])
    grounding = ground_claims(note_body, ev, tool_names, seq_metrics, experiment_plan) if ev else {
        "claims_total": 0, "claims_need_evidence": 0, "claims_grounded": 0,
        "grounding_score": 0, "ungrounded": [], "evidence_ids": [],
    }
    g_ok = grounding["grounding_score"] >= GROUNDING_TARGET * 100
    check(f"引用接地率 ≥ {int(GROUNDING_TARGET * 100)}%", g_ok,
          f"仅 {grounding['claims_grounded']}/{grounding['claims_need_evidence']} 条事实性主张有可核对引用",
          hard=False)

    rule_total = len(checks) or 1
    rule_score = 100 * sum(1 for c in checks if c["ok"]) / rule_total
    score = round(0.5 * rule_score + 0.5 * grounding["grounding_score"])
    # 通过门：无硬失败 且 总分 ≥ 90 且 接地率 ≥ 85%。只看"有没有硬失败"会让低分笔记也显示通过。
    passed = (not issues) and score >= 90 and grounding["grounding_score"] >= GROUNDING_TARGET * 100

    feedback: list[str] = []
    for i in issues:
        feedback.append(f"【必须修复】{i}")
    for w in warnings:
        feedback.append(f"【建议修复】{w}")
    if not issues and not passed:
        feedback.append(
            f"【门控未通过】总分 {score}/100（需 ≥ 90）或引用接地率 "
            f"{grounding['grounding_score']}%（需 ≥ {int(GROUNDING_TARGET * 100)}%）未达标，"
            "请按下列接地失败逐条改写或删除主张。"
        )
    for u in grounding["ungrounded"][:6]:
        feedback.append(
            f"【接地失败】主张「{u['text'][:60]}…」{u['reason']}；"
            "请改写为引用本轮命中文献中确实支持它的那一条，或删除该主张。"
        )

    return {
        "passed": passed,
        "score": score,
        "rule_score": round(rule_score),
        "grounding_score": grounding["grounding_score"],
        "grounding": grounding,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "cited_ids": cited,
        "repair_feedback": feedback,
        "summary": "校验通过" if passed else "校验未通过",
    }
