"""搜—读—算—写—辩—赛 控制环 / Control loop: search -> read -> compute -> write -> adjudicate -> tournament."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Settings, load_settings
from .ledger import Ledger, sha256_text
from .llm import SYSTEM_PROMPT, LLMError, OfflineLLM, _redact, build_llm, parse_tool_call
from .tools import load_corpus, run_tool
from .verifier import REQUIRED_SECTIONS, verify_note_claims
from .evidence import analyze as analyze_evidence
from .hypotheses import run as run_tournament
from .roles import TEAM_DISCLAIMER, role_for_phase, role_for_tool, role_summary
from .grounding import reground_note
from .repair import evidence_pack, failed_sections, parse_patch, sanitize_citations, splice_sections, split_sections, whitelist_rules

def _preview_args(args: dict[str, Any]) -> str:
    """时间线标题用的 args 预览：长字符串（尤其 119 aa 序列）截断为前 12 字 + 长度。

    完整 args 仍记在账本 step_record 里；这里只解决 UI 显示：不截断时整条序列是
    一个无断点的 119 字符长串，会把时间线卡片撑到 ~1156px 宽（900px 窗口下 193px
    的列被撑到 1157px），即用户报告的“文字溢出”。
    """
    short: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 40:
            short[k] = f"{v[:12]}…({len(v)} 字符)"
        else:
            short[k] = v
    return json.dumps(short, ensure_ascii=False)[:200]


EventFn = Callable[[dict[str, Any]], None]

PHASE_BY_TOOL = {
    "literature_search": "搜",
    "sequence_check": "算",
    "design_experiment": "设",
    "write_note": "写",
}
REPAIR_TARGET = 90          # 自修复目标分 / critic-repair target score
MAX_REPAIR_ROUNDS = 3
MAX_TOOL_RETRIES = 2       # 同一步工具失败后的最大重试次数 / retries per failing step
MAX_PATCH_SECTIONS = 1   # 单轮只重写最差的一个小节 / rewrite only the failing section


@dataclass
class AgentRunResult:
    mode: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    note_path: str | None = None
    score_trajectory: list[dict[str, Any]] = field(default_factory=list)
    repairs: int = 0
    tool_retries: int = 0
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    best_round: int = 0
    sanitize: dict[str, Any] = field(default_factory=dict)
    run_manifest_path: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    note_markdown: str = ""
    verify: dict[str, Any] | None = None
    evidence_report: dict[str, Any] = field(default_factory=dict)
    hypothesis_report: dict[str, Any] = field(default_factory=dict)
    literature_ids: list[str] = field(default_factory=list)
    literature_fallback: bool = False  # S1：本轮是否有兜底推荐命中
    literature_fallback_ids: list[str] = field(default_factory=list)
    roles: dict[str, Any] = field(default_factory=dict)  # P2-3：团队模式时按角色汇总的工具调用
    elapsed_s: float = 0.0
    goal: str = ""
    max_steps: int = 10
    success: bool = False
    error: str | None = None


class ScienceAgent:
    """科学智能体：意图 → 计划 → 工具 → 校验 → 沉淀。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        on_event: EventFn | None = None,
        verbose: bool = True,
        team: bool = False,  # P2-3：团队模式（单模型多角色编排），默认关闭、行为与现有一致
    ) -> None:
        self.settings = settings or load_settings()
        self.llm = build_llm(self.settings)
        self.corpus = load_corpus(self.settings.corpus_path)
        self.team = team
        self.corpus_sha = sha256_text(
            json.dumps(self.corpus, ensure_ascii=False, sort_keys=True)
        )[:16]
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event
        self.verbose = verbose
        self.evidence: list[dict[str, Any]] = []
        self.ledger = Ledger(
            mode="live" if self.settings.is_live else "offline",
            model=self.settings.model if self.settings.is_live else "OfflineLLM(deterministic)",
        )

    # ---------- 事件 / events ----------
    def _emit(self, kind: str, text: str, **extra: Any) -> dict[str, Any]:
        # P2-3：团队模式时按阶段自动打角色标签（不新增调用，只做可审计展示）
        if self.team and "role" not in extra:
            role = role_for_phase(str(extra.get("phase", "")))
            if role:
                extra["role"] = role
        ev = {"kind": kind, "text": text, "ts": time.time(), **extra}
        if self.verbose:
            icon = {
                "start": "▶", "plan": "🧠", "tool": "🛠", "tool_ok": "✓",
                "tool_fail": "✗", "verify": "🔍", "done": "✔", "error": "⚠", "repair": "🩹",
            }.get(kind, "·")
            print(f"{icon} {text}")
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 - UI 回调不应打断主流程
                pass
        return ev

    def _chat(self, messages: list[dict[str, str]], step: int, timeout: float | None = None,
              max_tokens: int | None = None, retries: int | None = None) -> str:
        """带计量的 LLM 调用 / metered LLM call."""
        t0 = time.time()
        kw: dict[str, Any] = {}
        if timeout:
            kw["timeout"] = timeout
        if max_tokens:
            kw["max_tokens"] = max_tokens
        if retries is not None:
            kw["retries"] = retries
        try:
            reply = self.llm.chat(messages, **kw)
        except LLMError:
            self.ledger.add_call(step=step, latency_s=round(time.time() - t0, 3), model=self.ledger.model, ok=False,
                                 prompt_sha256=sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                                 phase="repair" if step > getattr(self, "_write_step", 10**9) else "agent")
            raise
        usage = getattr(self.llm, "last_usage", {}) or {}
        self.ledger.add_call(
            step=step,
            latency_s=round(time.time() - t0, 3),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            model=self.ledger.model,
            prompt_sha256=sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
            phase="repair" if step > getattr(self, "_write_step", 10**9) else "agent",
        )
        return reply

    def run(self, user_goal: str, max_steps: int = 10, repair: bool = True) -> AgentRunResult:
        self._t0 = t0 = time.time()
        mode = "live" if self.settings.is_live else "offline"
        result = AgentRunResult(mode=mode)
        result.goal = user_goal
        result.max_steps = max_steps
        retry_counts: dict[str, int] = {}
        # S12：离线脚本 LLM 按用户目标参数化（靶点/序列）；LIVE 模式不走此分支
        if isinstance(self.llm, OfflineLLM):
            self.llm.configure(user_goal)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户目标：{user_goal}\n"
                    "请按 意图→计划→工具→校验→写作 推进。"
                    "可用工具：literature_search, sequence_check, design_experiment, write_note。"
                    "顺序建议：literature_search → sequence_check → design_experiment → write_note。"
                    "写笔记时，每条事实性主张后必须用 [Pxxx] 标注本轮命中的文献 ID。"
                ),
            },
        ]

        result.events.append(
            self._emit("start", f"模式 {mode} · 模型 {self.settings.model if mode == 'live' else 'OfflineLLM（确定性脚本）'}"
            f" · 目标：{user_goal}"
            + (" · 团队模式：检索员/计算员/评审员（单模型多角色编排）" if self.team else ""),
                       phase="意图", mode=mode)
        )

        for i in range(1, max_steps + 1):
            try:
                # 主循环显式传超时与重试上限：默认 120s/次、最多 1 次重试。
                # 不传时 LiveLLM 用客户端默认（曾为 300s × 2 次重试），
                # 接口挂住时单步可静默卡 ~15 分钟且界面只有一个静态 spinner，
                # 即用户报告的“莫名其妙卡住”。离线模式不受影响（OfflineLLM 忽略这些参数）。
                reply = self._chat(messages, i, max_tokens=self.settings.agent_max_tokens,
                                   timeout=self.settings.timeout, retries=1)
            except LLMError as exc:
                msg = (
                    f"LIVE 调用失败：{_redact(str(exc))}。"
                    "建议改用离线模式复现（python scripts/run_demo.py --mode offline，"
                    "或在 UI 左侧切换到 offline）。"
                )
                result.events.append(self._emit("error", msg, phase="错误"))
                result.error = _redact(str(exc))  # S17：失败信息过脱敏，不带出 Key
                result.elapsed_s = time.time() - t0
                # S11：失败也写账本，旧成功账本不得冒充本次运行
                self._write_failure_manifest(result, result.error)
                return result

            messages.append({"role": "assistant", "content": reply})
            tool_call = parse_tool_call(reply)
            step_record: dict[str, Any] = {"step": i, "assistant": reply}

            if not tool_call:
                preview = reply if len(reply) <= 400 else reply[:400] + "…"
                result.events.append(
                    self._emit("plan", f"[步骤 {i}] 规划 / 推理\n{preview}", phase="读", step=i, body=reply)
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "若计划已明确，请立刻输出一个工具调用 JSON；"
                            "若文献与校验已完成，请调用 write_note。"
                        ),
                    }
                )
                result.steps.append(step_record)
                continue

            name = str(tool_call.get("tool", ""))
            args = tool_call.get("args") or {}
            phase = PHASE_BY_TOOL.get(name, "工具")
            result.events.append(
                self._emit(
                    "tool",
                    f"[步骤 {i}] 调用工具 {name} · args={_preview_args(args)}",
                    phase=phase, step=i, tool=name, args=args,
                )
            )

            tool_result = run_tool(
                name, args, corpus=self.corpus, outputs_dir=self.settings.outputs_dir
            )
            self.ledger.tool_calls += 1
            step_record.update(
                tool=name, args=args, tool_ok=tool_result.ok,
                tool_summary=tool_result.summary, tool_data=tool_result.data,
            )
            # P2-3：团队模式时按工具打角色标签（默认关闭，不写该字段）
            if self.team:
                step_record["role"] = role_for_tool(name)
            result.steps.append(step_record)
            result.events.append(
                self._emit(
                    "tool_ok" if tool_result.ok else "tool_fail",
                    tool_result.summary, phase=phase, step=i, tool=name,
                    data=tool_result.data,
                )
            )

            if name == "literature_search" and tool_result.ok:
                # S1：兜底命中另册记录（仍进白名单以便笔记引用，但账本与事件显性标注 fallback）
                if tool_result.data.get("fallback"):
                    result.literature_fallback = True
                    fb_ids = [str(h.get("id")) for h in tool_result.data.get("hits", []) if h.get("fallback")]
                    result.literature_fallback_ids.extend(i for i in fb_ids if i not in result.literature_fallback_ids)
                    result.events.append(self._emit(
                        "error",
                        f"⚠ 文献检索无精确命中，以下 {len(fb_ids)} 篇为按年份默认推荐：{', '.join(fb_ids)}，不可作为检索证据引用",
                        phase="搜",
                    ))
                # 多轮检索：合并去重，避免后一次覆盖白名单与接地基准
                seen = set(result.literature_ids)
                for h in tool_result.data.get("hits", []) or []:
                    hid = h.get("id")
                    if not hid:
                        continue
                    hid = str(hid)
                    if hid in seen:
                        continue
                    seen.add(hid)
                    result.literature_ids.append(hid)
                    self.evidence.append(h)
                tool_result.summary += (
                    "\n\n【证据片段（写笔记时只能引用这些）】\n" + evidence_pack(self.evidence)
                    + "\n\n" + whitelist_rules(result.literature_ids)
                )

            if name == "write_note" and tool_result.ok:
                return self._finish(result, tool_result, messages, step_record, i, repair)

            if not tool_result.ok:
                used = retry_counts.get(name, 0)
                step_record["retry_index"] = used
                if used < MAX_TOOL_RETRIES:
                    retry_counts[name] = used + 1
                    self.ledger.tool_retries += 1
                    ev = {"step": i, "tool": name, "attempt": used + 1,
                          "reason": tool_result.summary[:200],
                          "args": {k: str(x)[:120] for k, x in (args or {}).items()}}
                    self.ledger.retry_events.append(ev)
                    result.retry_events.append(ev)
                    result.tool_retries += 1
                    result.events.append(self._emit(
                        "repair",
                        f"[步骤 {i}] 工具 {name} 失败，第 {used + 1}/{MAX_TOOL_RETRIES} 次重试",
                        phase=phase, step=i, tool=name,
                    ))
                    messages.append({
                        "role": "user",
                        "content": (
                            "【工具失败】\n"
                            + json.dumps({"tool": name, "ok": False, "summary": tool_result.summary,
                                          "data": tool_result.data, "args": args},
                                         ensure_ascii=False, indent=2)
                            + f"\n请用修正后的参数重新调用同一个工具 {name}（数值参数只给纯数字，"
                              f"不要带单位或 pH 前缀）。这是第 {used + 1} 次重试，最多 {MAX_TOOL_RETRIES} 次。"
                        ),
                    })
                    continue
                result.events.append(self._emit(
                    "tool_fail",
                    f"[步骤 {i}] 工具 {name} 连续失败 {MAX_TOOL_RETRIES} 次，跳过该步骤继续后续流程",
                    phase=phase, step=i, tool=name,
                ))
                messages.append({
                    "role": "user",
                    "content": (
                        f"工具 {name} 已重试 {MAX_TOOL_RETRIES} 次仍失败：{tool_result.summary}\n"
                        "请不要再调用该工具，改为调用流程中的下一个工具并在笔记中如实记录该步骤失败。"
                    ),
                })
                result.steps[-1]["skipped_after_retries"] = True
                continue
            retry_counts[name] = 0

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "【工具结果】\n"
                        + json.dumps(
                            {
                                "tool": tool_result.name,
                                "ok": tool_result.ok,
                                "summary": tool_result.summary,
                                "data": tool_result.data,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n请根据结果继续：尚未校验序列则调用 sequence_check；"
                        "已校验但未设计实验则调用 design_experiment；"
                        "两者齐备后调用 write_note 写完整实验笔记"
                        "（每条事实性主张后标注 [Pxxx] 引用）。"
                    ),
                }
            )

        result.elapsed_s = time.time() - t0
        result.error = "达到最大步数仍未写出笔记"
        result.events.append(self._emit("error", f"⚠ {result.error}（max_steps={max_steps}）", phase="错误"))
        # S11：步数耗尽也写失败账本，旧成功账本不得冒充本次运行
        self._write_failure_manifest(result, result.error)
        return result

    def _write_failure_manifest(self, result: AgentRunResult, error: str) -> None:
        """S11：失败分支落盘失败账本（success=false + error），覆盖旧的
        last_run_manifest.json 并另存 runs/<timestamp>-<mode>-failed.json。"""
        self.ledger.inputs = {
            "goal_sha256": sha256_text(result.goal),
            "corpus_sha256": self.corpus_sha,
            "corpus_papers": len(self.corpus),
            "max_steps": result.max_steps,
            "temperature": 0.2,
            "seed": self.settings.seed,
            "seed_policy": "offline=deterministic script; live=seed passed to API (provider may ignore)",
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        }
        self.ledger.outputs = {
            "success": False,
            "error": error,
            "note_sha256": None,
            "note_path": None,
            "verify_score": None,
            "grounding_score": None,
            "repair_rounds": result.repairs,
            "tool_retries": result.tool_retries,
            "score_trajectory": result.score_trajectory,
        }
        result.manifest = self.ledger.manifest()
        (self.settings.outputs_dir / "last_run_manifest.json").write_text(
            json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{self.ledger.mode}-failed"
        rp = self.ledger.dump(self.settings.outputs_dir / "runs" / f"{run_id}.json")
        result.run_manifest_path = str(rp)

    # ---------- 收尾：校验 + 自修复环 + 账本 ----------
    def _seq_metrics(self, result: AgentRunResult) -> dict[str, Any] | None:
        """从本轮 sequence_check 步骤提取理化指标（供 TOOL 徽章数值核对）。"""
        step = next((s for s in reversed(result.steps) if s.get("tool") == "sequence_check"), None)
        data = (step or {}).get("tool_data") or {}
        if not data:
            return None
        metrics: dict[str, Any] = {"length": data.get("length"), "assay_ph": data.get("assay_ph")}
        for m in data.get("metrics", []) or []:
            if m.get("name"):
                metrics[m["name"]] = m.get("value")
        return metrics

    def _experiment_plan(self, result: AgentRunResult) -> dict[str, Any] | None:
        """从本轮 design_experiment 步骤提取实验设计（供 TOOL 徽章实验名核对）。"""
        step = next((s for s in reversed(result.steps) if s.get("tool") == "design_experiment"), None)
        data = (step or {}).get("tool_data") or {}
        return data or None

    def _verify(self, body: str, result: AgentRunResult) -> dict[str, Any]:
        seq_step = next((s for s in reversed(result.steps) if s.get("tool") == "sequence_check"), None)
        return verify_note_claims(
            body,
            literature_ids=result.literature_ids or None,
            sequence_ok=bool(seq_step.get("tool_ok")) if seq_step else None,
            evidence=self.evidence,
            physchem_ok=(seq_step or {}).get("tool_data", {}).get("ok"),
            tool_names=[s.get("tool") for s in result.steps if s.get("tool")],
            seq_metrics=self._seq_metrics(result),
            experiment_plan=self._experiment_plan(result),
        )

    def _finish(
        self,
        result: AgentRunResult,
        tool_result: Any,
        messages: list[dict[str, str]],
        step_record: dict[str, Any],
        step: int,
        repair: bool,
    ) -> AgentRunResult:
        self._write_step = step
        note_path = tool_result.data.get("latest") or tool_result.data.get("path")
        body = Path(note_path).read_text(encoding="utf-8")
        wl = [str(h.get("id")) for h in self.evidence if h.get("id")]
        body, san0 = sanitize_citations(body, wl)
        tools_run = [s.get("tool") for s in result.steps if s.get("tool")]
        seq_metrics = self._seq_metrics(result)
        exp_plan = self._experiment_plan(result)
        body, rg0 = reground_note(body, self.evidence, tools_run, seq_metrics=seq_metrics, experiment_plan=exp_plan)
        san0 = {**san0, "reground": rg0}
        Path(note_path).write_text(body, encoding="utf-8")
        result.sanitize = san0
        self._emit("verify", f"引用后处理：接地 {rg0['grounding_before']}% → {rg0['grounding_after']}%"
                   f"（重匹配 {len(rg0['rematched'])} 句，删除 {len(rg0['dropped'])} 句）", phase="校验")
        v = self._verify(body, result)
        result.note_markdown = body
        result.note_path = note_path
        result.score_trajectory.append(
            {"round": 0, "score": v["score"], "rule": v["rule_score"], "grounding": v["grounding_score"],
             "grounding_raw": rg0["grounding_before"],
             "failed_checks": [c["name"] for c in v.get("checks", []) if not c.get("ok")], "sanitized": san0}
        )
        self._emit(
            "verify",
            f"校验器 R0：{v['score']}/100（规则 {v['rule_score']} · 接地 {v['grounding_score']}）",
            phase="校验", data=v,
        )

        # ---- 补丁式自修复环 / patch-shaped critic-repair loop ----
        rounds = 0
        best_v, best_body, best_path = v, body, note_path
        title = (tool_result.data.get("title") or "lab_note")
        ids = [str(h.get("id")) for h in self.evidence if h.get("id")]
        budget = float(self.settings.repair_budget)
        t_budget = time.time()
        stop_reason = "target_reached" if best_v["score"] >= REPAIR_TARGET else ("repair_disabled" if not repair else "")
        last_round_s, last_failed = 0.0, False
        while repair and best_v["score"] < REPAIR_TARGET:
            if rounds >= MAX_REPAIR_ROUNDS:
                stop_reason = "call_failure" if last_failed else "rounds_exhausted"; break
            remaining = budget - (time.time() - t_budget)
            # 预估本轮耗时（上一轮耗时），不够就提前停止 / stop early if the next round likely won't fit
            if remaining < max(5.0, last_round_s):
                stop_reason = "budget_exhausted"
                self._emit("repair", f"[自修复] 预算 {budget:.0f}s 用尽，保留最优版本 {best_v['score']}/100", phase="修")
                break
            rounds += 1
            failed = [f"{c['name']}：{c.get('detail','')}" for c in v.get("checks", []) if not c.get("ok")]
            fb = "\n".join(f"- {x}" for x in (failed + list(v.get("repair_feedback", [])))[:12])
            targets = (failed_sections(body, v, REQUIRED_SECTIONS) or [h for h, _ in split_sections(body) if h][:2])[:MAX_PATCH_SECTIONS]
            cur = dict(split_sections(body))
            sec_txt = "\n\n".join(f"{h}\n{cur.get(h, '（缺失，请新写）')}" for h in targets)
            self._emit("repair", f"[自修复 R{rounds}] 当前 {v['score']}/100 < {REPAIR_TARGET}，只重写 {len(targets)} 个小节："
                       + "、".join(targets), phase="修")
            rmsgs = [
                {"role": "system", "content": "你是科研笔记修订器。只输出被要求的 Markdown 小节（以 '## 标题' 开头），不要输出 JSON、解释或其他小节。"},
                {"role": "user", "content": (
                    f"【证据片段】\n{evidence_pack(self.evidence)}\n\n{whitelist_rules(ids)}\n\n"
                    f"【校验器反馈】得分 {v['score']}/100（规则 {v['rule_score']}，接地 {v['grounding_score']}）：\n{fb}\n\n"
                    f"【需重写的小节】（保持同样的 '## ' 标题，逐句句末加引用；每节 ≤ 220 字，只保留有引用的要点句，删掉无法引用的句子）\n{sec_txt}"
                )},
            ]
            t_r = time.time()
            try:
                reply = self._chat(rmsgs, step + rounds, timeout=max(5.0, min(self.settings.repair_timeout, remaining)),
                                   max_tokens=self.settings.repair_max_tokens, retries=0)
            except LLMError as exc:
                self._emit("error", f"自修复调用失败：{exc}", phase="错误")
                result.score_trajectory.append({"round": rounds, "score": None, "error": str(exc)[:160],
                                                "latency_s": round(time.time() - t_r, 2)})
                last_round_s, last_failed = time.time() - t_r, True
                continue
            last_failed = False
            patch = parse_patch(reply)
            if not patch:
                result.score_trajectory.append({"round": rounds, "score": None, "error": "空补丁",
                                                "latency_s": round(time.time() - t_r, 2)})
                last_round_s = time.time() - t_r
                continue
            nbody, used, dropped = splice_sections(body, patch, allowed_new=targets)
            nbody, san = sanitize_citations(nbody, ids)
            nbody, rg = reground_note(nbody, self.evidence, tools_run, seq_metrics=seq_metrics, experiment_plan=exp_plan)
            san = {**san, "reground": {k: (len(x) if isinstance(x, list) else x) for k, x in rg.items()},
                   "splice_dropped": dropped}
            tr = run_tool("write_note", {"title": title, "body_markdown": nbody}, corpus=self.corpus,
                          outputs_dir=self.settings.outputs_dir)
            self.ledger.tool_calls += 1
            note_path = tr.data.get("latest") or tr.data.get("path")
            body = Path(note_path).read_text(encoding="utf-8")
            nv = self._verify(body, result)
            result.score_trajectory.append(
                {"round": rounds, "score": nv["score"], "rule": nv["rule_score"],
                 "grounding": nv["grounding_score"], "grounding_raw": rg["grounding_before"], "patched_sections": used,
                 "dropped_sections": dropped,
                 "latency_s": round(time.time() - t_r, 2), "sanitized": san,
                 "failed_checks": [c["name"] for c in nv.get("checks", []) if not c.get("ok")]}
            )
            self._emit("verify", f"校验器 R{rounds}：{nv['score']}/100（规则 {nv['rule_score']} · 接地 {nv['grounding_score']}）"
                       f" · Δ{nv['score'] - best_v['score']:+d} · {time.time() - t_r:.1f}s", phase="校验", data=nv)
            last_round_s = time.time() - t_r
            if nv["score"] > best_v["score"]:
                best_v, best_body, best_path = nv, body, note_path
            else:
                self._emit("repair", f"[自修复 R{rounds}] 未超过最优（{nv['score']} ≤ {best_v['score']}），保留最优版本", phase="修")
            v = nv
        if not stop_reason:
            stop_reason = "target_reached" if best_v["score"] >= REPAIR_TARGET else ("call_failure" if last_failed else "rounds_exhausted")
        result.sanitize["repair_stop_reason"] = stop_reason
        result.sanitize["repair_phase_s"] = round(time.time() - t_budget, 2)

        v, body, note_path = best_v, best_body, best_path
        if note_path:
            Path(note_path).write_text(body, encoding="utf-8")

        # ---- 证据辩证层：支撑强度分级 + 过度表述降级 + 文献分歧检测 ----
        evidence_rewrote = False
        try:
            ev_report = analyze_evidence(body, self.evidence, self.corpus,
                                         tool_names=tools_run, seq_metrics=seq_metrics, experiment_plan=exp_plan)
            if ev_report["note_markdown"] != body:
                body = ev_report["note_markdown"]
                evidence_rewrote = True
                if note_path:
                    Path(note_path).write_text(body, encoding="utf-8")
            cnt = ev_report["support"]["counts"]
            dcnt = ev_report["disagreements"]["counts"]
            self._emit(
                "verify",
                f"证据辩证：直接 {cnt['direct']} · 间接 {cnt['indirect']} · 过度表述 {cnt['overstated']}（已降级）"
                f" · 无支撑 {cnt['unsupported']}；文献分歧 实质 {dcnt['real']} / 条件差异 {dcnt['conditions']}"
                f" / 口径差异 {dcnt['definitions']}",
                phase="校验", data=ev_report,
            )
        except Exception as exc:  # 证据层失败不应拖垮整次运行
            ev_report = {"error": str(exc)}
            self._emit("verify", f"⚠ 证据辩证层未完成：{exc}", phase="校验")
        result.evidence_report = ev_report

        # 仅当辩证层改写了终稿时重验：若改差了（分低于修复环最优）则回退到最优版，
        # 避免直接采用改差的版本；best_round 在终稿确定后再算。
        if evidence_rewrote and note_path:
            v2 = self._verify(body, result)
            if v2["score"] >= best_v["score"]:
                v = v2
                best_v, best_body, best_path = v2, body, note_path
                adopted = True
            else:
                body, note_path = best_body, best_path
                v = best_v
                Path(note_path).write_text(body, encoding="utf-8")
                adopted = False
                result.sanitize["evidence_rollback"] = {
                    "post_evidence_score": v2.get("score"),
                    "best_score": best_v["score"],
                    "softened_count": len((ev_report.get("support", {}) or {}).get("softened", [])),
                }
                # 证据报告按回退后的终稿重算，保持与正文一致；未被采纳的降级不计入终稿报告
                try:
                    ev_report = analyze_evidence(body, self.evidence, self.corpus,
                                                 tool_names=tools_run, seq_metrics=seq_metrics, experiment_plan=exp_plan)
                    ev_report["note_markdown"] = body
                    ev_report["support"]["softened"] = []
                    result.evidence_report = ev_report
                except Exception:  # noqa: BLE001 - 重算失败时保留原报告，不拖垮运行
                    pass
            result.score_trajectory.append({
                "round": "post_evidence", "score": v2.get("score"),
                "grounding": v2.get("grounding_score"), "passed": v2.get("passed"),
                "adopted": adopted,
            })

        # best_round 在终稿确定后再算：回退时指向修复轮次，采用时可能指向 post_evidence
        result.best_round = next((t["round"] for t in result.score_trajectory if t.get("score") == v["score"]), 0)

        # ---- 假设锦标赛：优先本轮命中；不足时回退全库（保离线可复现）----
        # S6（方案 B）：保留全库回退，但显性标注池来源——冠军 support/counter 可能不在本轮引用内。
        try:
            tourney_pool = self.evidence if len(self.evidence) >= 8 else self.corpus
            pool_source = "evidence" if len(self.evidence) >= 8 else "corpus"
            hyp_report = run_tournament(tourney_pool, limit=6)
            hyp_report["tournament_pool"] = pool_source
            hyp_report["pool_ids"] = [str(p.get("id")) for p in tourney_pool if p.get("id")]
            hyp_report["pool_source"] = (
                "本轮检索命中（≥8 篇）" if pool_source == "evidence"
                else "全库分析（本轮命中不足 8 篇，非本轮检索证据；冠军 support/counter 可能不在本轮引用内）"
            )
            top = (hyp_report.get("hypotheses") or [{}])[0]
            self._emit(
                "verify",
                f"假设锦标赛[{hyp_report['pool_source']}]：{len(hyp_report.get('hypotheses', []))} 个假设 · {hyp_report.get('rounds', 0)} 场对决 · "
                f"冠军 {top.get('id', '—')}（Elo {top.get('elo', '—')} · 存活分 {top.get('survival', '—')}）",
                phase="校验", data=hyp_report,
            )
        except Exception as exc:  # noqa: BLE001
            hyp_report = {"error": str(exc)}
            self._emit("verify", f"⚠ 假设锦标赛未完成：{exc}", phase="校验")
        result.hypothesis_report = hyp_report

        result.repairs = rounds
        result.note_path = note_path
        result.note_markdown = body
        result.verify = v
        result.success = bool(v.get("passed"))  # 与校验器 passed 门挂钩，不再恒为 True
        result.sanitize["verify_passed"] = bool(v.get("passed"))
        step_record["verify"] = v
        # P2-3：团队模式时账本记角色汇总（只统计已有步骤，不新编数据）
        if self.team:
            result.roles = role_summary(result.steps)

        # ---- 可复现账本 / reproducibility manifest ----
        self.ledger.inputs = {
            "goal_sha256": sha256_text(result.goal),
            "corpus_sha256": self.corpus_sha,
            "corpus_papers": len(self.corpus),
            "max_steps": result.max_steps,
            "temperature": 0.2,
            "seed": self.settings.seed,
            "seed_policy": "offline=deterministic script; live=seed passed to API (provider may ignore)",
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "base_url": self.settings.base_url if self.settings.is_live else None,
            "repair_target": REPAIR_TARGET, "max_repair_rounds": MAX_REPAIR_ROUNDS,
            "repair_budget_s": self.settings.repair_budget, "repair_max_tokens": self.settings.repair_max_tokens,
            "agent_max_tokens": self.settings.agent_max_tokens, "thinking_mode": self.settings.thinking,
            "literature_fallback": result.literature_fallback,  # S1：本轮是否含兜底推荐命中
            "literature_fallback_ids": result.literature_fallback_ids,
        }
        # 去掉时间戳注释后再哈希，保证"同输入 → 同哈希" / hash the content, not the timestamp
        canonical = re.sub(r"<!--.*?-->\s*", "", body, flags=re.S).strip()
        # P0-8：账本里的 note_path 写相对路径（相对包根），不再含 /home/ 或 /workspace/ 绝对前缀
        try:
            note_rel = str(Path(note_path).relative_to(self.settings.outputs_dir.parent))
        except (ValueError, TypeError):
            note_rel = str(note_path)
        self.ledger.outputs = {
            "note_sha256": sha256_text(canonical),
            "note_sha256_scope": "正文（已剔除生成时间注释）",
            "note_path": note_rel,
            "verify_score": v["score"],
            "grounding_score": v["grounding_score"],
            "repair_rounds": rounds,
            "tool_retries": result.tool_retries,
            "repair_stop_reason": stop_reason,
            "repair_phase_s": result.sanitize["repair_phase_s"],
            "repair_budget_s": budget,
            "grounding_before_postprocess": rg0["grounding_before"],
            "grounding_after_postprocess": rg0["grounding_after"],
            "best_round": result.best_round,
            "score_trajectory": result.score_trajectory,
            "evidence_support": (result.evidence_report or {}).get("support", {}).get("counts"),
            "evidence_support_score": (result.evidence_report or {}).get("support", {}).get("support_score"),
            "evidence_softened": len((result.evidence_report or {}).get("support", {}).get("softened", [])),
            "evidence_disagreements": (result.evidence_report or {}).get("disagreements", {}).get("counts"),
            "evidence_confidence": (result.evidence_report or {}).get("confidence"),
            "evidence_error": (result.evidence_report or {}).get("error"),
            "hypotheses": (result.hypothesis_report or {}).get("hypotheses"),
            "hypothesis_rounds": (result.hypothesis_report or {}).get("rounds"),
            "tournament_pool": (result.hypothesis_report or {}).get("tournament_pool"),
            "tournament_pool_ids": (result.hypothesis_report or {}).get("pool_ids"),
            "tournament_pool_source": (result.hypothesis_report or {}).get("pool_source"),
            "next_experiment": (result.hypothesis_report or {}).get("next_experiment"),
            "hypothesis_error": (result.hypothesis_report or {}).get("error"),
        }
        # P2-3：团队模式时账本 outputs 记角色汇总与编排口径（默认关闭，不写该字段）
        if self.team:
            self.ledger.outputs["roles"] = result.roles
            self.ledger.outputs["team_mode"] = TEAM_DISCLAIMER
        result.manifest = self.ledger.manifest()
        (self.settings.outputs_dir / "last_run_manifest.json").write_text(
            json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{self.ledger.mode}-{self.ledger.outputs['note_sha256'][:8]}"
        rp = self.ledger.dump(self.settings.outputs_dir / "runs" / f"{run_id}.json")
        result.run_manifest_path = str(rp)
        result.elapsed_s = time.time() - self._t0
        self._emit(
            "done",
            f"闭环完成：{v['score']}/100 · 修复 {rounds} 轮 · {result.elapsed_s:.1f}s · {note_path}",
            phase="写",
        )
        return result
