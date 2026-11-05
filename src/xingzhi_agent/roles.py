"""多智能体分工：单模型多角色编排（P2-3）。

轻量角色编排，不重构控制环，不新增 LLM 调用：
- 角色 = 提示词 + 工具子集 + 事件标签；现有 pipeline 的每一步按所用工具/阶段打上角色标签，
  记入账本（ledger.outputs["roles"]）与 UI 时间线（事件的 role 字段）。
- 角色只做编排与可审计展示：同一基座模型按角色切换提示词与工具子集。
  文档、UI 与账本必须如实写“单模型多角色编排”，不得声称是多个独立智能体。

角色分工（与 plan.md §4 P2-3 一致）：
- 检索员：literature_search / parse_figure（搜文献、读图注）；
- 计算员：sequence_check / design_experiment（算理化指标、设计实验）；
- 评审员：write_note（执笔）+ 校验 / 证据辩证 / 假设锦标赛 / 自修复（审稿）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TEAM_DISCLAIMER = (
    "单模型多角色编排：同一基座模型按角色切换提示词与工具子集，"
    "不是多个独立智能体；角色只做编排与可审计展示，不新增 LLM 调用。"
)


@dataclass(frozen=True)
class Role:
    """一个角色：提示词 + 工具子集 + 事件标签。"""
    id: str  # retriever / calculator / reviewer（账本与事件用）
    name: str  # 检索员 / 计算员 / 评审员（展示用）
    prompt: str  # 该角色的系统提示词片段（接在主 SYSTEM_PROMPT 之后，不新增调用）
    tools: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    description: str = ""


RETRIEVER = Role(
    id="retriever",
    name="检索员",
    prompt="你是检索员：只用 literature_search / parse_figure 找证据；"
           "每条主张必须带本轮命中的 [Pxxx] 或 [Pxxx:FigN] 引用，无精确命中时明确标注兜底推荐，不得编造来源。",
    tools=("literature_search", "parse_figure"),
    phases=("搜", "读"),
    description="搜文献、读图注：产出带引用的证据片段。",
)

CALCULATOR = Role(
    id="calculator",
    name="计算员",
    prompt="你是计算员：只用 sequence_check / design_experiment 做可重算的数值工作；"
           "所有数值必须来自工具输出原文，assay_ph 在 [0, 14] 内，"
           "未知 molecule_type 时明确报错，不得静默按默认值算。",
    tools=("sequence_check", "design_experiment"),
    phases=("算", "设"),
    description="算理化指标、设计实验：产出可重算的数值与判据。",
)

REVIEWER = Role(
    id="reviewer",
    name="评审员",
    prompt="你是评审员：执笔写实验笔记，并组织校验、证据辩证、假设锦标赛与自修复；"
           "只保留有引用支撑的主张，过度表述降级，方向相反的引用不得作为支撑，"
           "分数未达目标时只重写未通过的小节。",
    tools=("write_note",),
    phases=("写", "校验", "修"),
    description="执笔、校验、辩证、锦标赛：产出可验证的实验笔记。",
)

ROLES: dict[str, Role] = {r.id: r for r in (RETRIEVER, CALCULATOR, REVIEWER)}

# 工具 → 角色（步骤打标签用）；阶段 → 角色（事件打标签用）
ROLE_BY_TOOL: dict[str, str] = {
    tool: role.id for role in ROLES.values() for tool in role.tools
}
ROLE_BY_PHASE: dict[str, str] = {
    phase: role.id for role in ROLES.values() for phase in role.phases
}


def role_for_tool(tool_name: str) -> str:
    """工具名 → 角色 id；未登记的工具返回 ''（不打标签，不断言）。"""
    return ROLE_BY_TOOL.get((tool_name or "").strip(), "")


def role_for_phase(phase: str) -> str:
    """阶段名 → 角色 id；未登记的阶段返回 ''。"""
    return ROLE_BY_PHASE.get((phase or "").strip(), "")


def role_name(role_id: str) -> str:
    """角色 id → 中文名；未知 id 原样返回。"""
    role = ROLES.get(role_id)
    return role.name if role else role_id


def role_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """按角色汇总工具调用（供账本 outputs['roles'] 与 UI 团队面板使用）。

    只统计已有步骤字段，不新编数据；未打标签的步骤记入 unassigned，不丢数。
    """
    per_role: dict[str, dict[str, Any]] = {
        r.id: {"role": r.id, "name": r.name, "tool_calls": 0, "tools": [],
               "ok": 0, "failed": 0, "description": r.description}
        for r in ROLES.values()
    }
    unassigned = 0
    for s in steps:
        tool = s.get("tool")
        if not tool:
            continue
        rid = role_for_tool(str(tool))
        if not rid:
            unassigned += 1
            continue
        entry = per_role[rid]
        entry["tool_calls"] += 1
        if tool not in entry["tools"]:
            entry["tools"].append(tool)
        if s.get("tool_ok") is False:
            entry["failed"] += 1
        else:
            entry["ok"] += 1
    return {
        "mode": "单模型多角色编排",
        "disclaimer": TEAM_DISCLAIMER,
        "roles": [per_role[r.id] for r in ROLES.values()],
        "unassigned_tool_calls": unassigned,
    }
