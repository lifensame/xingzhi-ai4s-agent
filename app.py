#!/usr/bin/env python3
"""兴智杯 · AI4S 科学智能体 —— 可视化控制台 / Streamlit console.

运行 / Run:  streamlit run app.py        或  python scripts/run_ui.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ui.theme import CSS  # noqa: E402
from ui.panels import empty_state, evidence_panel, hypothesis_panel, kpi_row, metric_cards, newest_run, run_ledger  # noqa: E402

st.set_page_config(
    page_title="兴智杯 · AI4S 科学智能体",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

DEFAULT_GOAL = (
    "请围绕 EGFR 靶点的 VHH（纳米抗体）做一轮文献预研：检索可表达性与 pH 相关要点，"
    "对示例序列做长度窗校验，并写一份可复现的 Markdown 实验笔记。"
)
PHASES = [("搜", "文献检索"), ("读", "要点蒸馏"), ("算", "理化校验"), ("设", "实验设计"), ("修", "自修复"), ("写", "笔记沉淀")]
ICONS = {"start": "🎯", "plan": "🧠", "tool": "🛠", "tool_ok": "✓", "tool_fail": "✗",
         "verify": "🔍", "done": "✔", "error": "⚠", "repair": "🩹"}


def esc(text: str) -> str:
    return html.escape(str(text))


def hero(mode: str, model: str, reached: set[str]) -> None:
    chips = "".join(
        f'<span class="xz-chip {"on" if p in reached else ""}">{p} · {n}</span>'
        for p, n in PHASES
    )
    st.markdown(
        f"""
<div class="xz-hero">
  <h1>兴智杯 · AI4S 科学智能体</h1>
  <div class="xz-sub">
    <b>搜 — 读 — 算 — 写 — 辩 — 赛</b> 闭环：本地文献检索 → 要点蒸馏 → 序列约束校验 → 可复现实验笔记 → 证据辩证 → 假设锦标赛。<br>
    当前模式 <b>{esc(mode)}</b> · 模型 <b>{esc(model)}</b> ·
    基座可对接书生（InternLM）OpenAI 兼容接口；离线模式无需任何密钥即可复现。
  </div>
  <div class="xz-chips">{chips}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def card(title: str, detail: str = "", dot: str = "·", state: str = "") -> str:
    d = f'<div class="d">{esc(detail)}</div>' if detail else ""
    return (
        f'<div class="xz-card"><div class="xz-step">'
        f'<div class="xz-dot {state}">{esc(dot)}</div>'
        f'<div style="flex:1;min-width:0"><div class="t">{esc(title)}</div>{d}</div>'
        f"</div></div>"
    )


def render_trajectory(traj: list[dict]) -> None:
    import altair as alt
    import pandas as pd

    rows = []
    for t in traj:
        for key, lab in (("score", "总分"), ("rule", "规则分"), ("grounding", "接地分")):
            if t.get(key) is not None:
                rows.append({"轮次": f"R{t['round']}", "指标": lab, "分数": t[key]})
    if rows:
        df = pd.DataFrame(rows)
        base = alt.Chart(df).encode(
            x=alt.X("轮次:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("分数:Q", scale=alt.Scale(domain=[0, 108], clamp=False), title=None),
            color=alt.Color("指标:N", scale=alt.Scale(domain=["总分", "规则分", "接地分"],
                            range=["#6EE7B7", "#94A3B8", "#475569"]), legend=alt.Legend(orient="top", title=None)),
        )
        goal = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(strokeDash=[5, 4], color="#6EE7B7", opacity=0.5).encode(y="y:Q")
        # 始终折线 + 点：单轮时也是同一种图形语言 / always a line chart, even with one round
        lines = base.mark_line(strokeWidth=2.5)
        if df["轮次"].nunique() == 1:  # 单轮：横向错开三个点，避免重叠 / spread points when only one round
            pts = base.mark_point(size=110, filled=True, opacity=1).encode(xOffset="指标:N")
            labels = base.mark_text(dy=-14, fontSize=12, color="#E2E8F0").encode(text="分数:Q", xOffset="指标:N")
        else:
            pts = base.mark_point(size=90, filled=True, opacity=1)
            labels = base.mark_text(dy=-12, fontSize=12, color="#E2E8F0").encode(text="分数:Q")
        chart = (lines + pts + labels + goal).properties(height=260)
        st.altair_chart(chart.configure(background="transparent").configure_axis(
            gridColor="#1E293B", labelColor="#8FA3B8", domainColor="#1E293B").configure_view(strokeOpacity=0)
            .configure_legend(labelColor="#CBD5E1"), use_container_width=True)
    table = [{"轮次": f"R{t['round']}", "总分": t.get("score") if t.get("score") is not None else "失败",
              "接地": t.get("grounding", "—"), "耗时 s": t.get("latency_s", "—"),
              "重写小节": "、".join(t.get("patched_sections") or []) or ("初稿" if t["round"] == 0 else "—"),
              "备注": t.get("error") or "、".join(t.get("failed_checks") or []) or "全部通过"} for t in traj]
    st.dataframe(table, hide_index=True, use_container_width=True)
    ok = [t for t in traj if t.get("score") is not None]
    if ok:
        best = max(ok, key=lambda t: t["score"])
        st.caption(f"R0 {ok[0]['score']} → 最优 R{best['round']} {best['score']}（Δ{best['score'] - ok[0]['score']:+d}）；"
                   "虚线为 90 分目标。修复只重写未通过的小节，再确定性拼回原文。")


def render_ledger_steps(man: dict) -> None:
    steps = man.get("steps") or []
    if not steps:
        return
    st.dataframe([{"步": s["step"], "阶段": "修复" if s.get("phase") == "repair" else "主环",
                   "状态": "✅" if s.get("ok", True) else "❌ 失败", "延迟 s": s["latency_s"],
                   "prompt tok": s.get("prompt_tokens", 0), "completion tok": s.get("completion_tokens", 0),
                   "prompt sha256": (s.get("prompt_sha256") or "")[:12]} for s in steps],
                 hide_index=True, use_container_width=True)


def render_run_history() -> None:
    allr = sorted((ROOT / "outputs" / "runs").glob("*.json"), reverse=True)
    runs = [p for p in allr if "-live-" in p.name][:8] + [p for p in allr if "-offline-" in p.name][:2]
    if not runs:
        return
    st.markdown("#### 🗂 运行档案 / Run ledger history")
    rows = []
    for p in runs:
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tr = (m.get("outputs") or {}).get("score_trajectory") or []
        rows.append({"运行": p.stem, "模式": m.get("mode"), "模型": m.get("model"),
                     "轨迹": " → ".join(str(t.get("score") if t.get("score") is not None else "×") for t in tr),
                     "最终": (m.get("outputs") or {}).get("verify_score"),
                     "接地": (m.get("outputs") or {}).get("grounding_score"),
                     "失败调用": sum(1 for s in m.get("steps", []) if not s.get("ok", True)),
                     "token": m.get("total_tokens"), "seed": (m.get("inputs") or {}).get("seed")})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    pick = st.selectbox("查看某次运行的评分轨迹", [r["运行"] for r in rows], index=0)
    m = json.loads((ROOT / "outputs" / "runs" / f"{pick}.json").read_text(encoding="utf-8"))
    tr = (m.get("outputs") or {}).get("score_trajectory") or []
    if tr:
        render_trajectory(tr)


def render_event(ev: dict) -> str:
    kind = ev.get("kind", "")
    icon = ICONS.get(kind, "·")
    state = "ok" if kind in {"tool_ok", "done", "verify"} else "err" if kind in {"tool_fail", "error"} else ""
    phase = ev.get("phase", "")
    # P2-3：团队模式时事件带 role 标签，时间线显示为 [阶段·角色中文名]
    role = ev.get("role", "")
    if role:
        from xingzhi_agent.roles import role_name as _role_name  # noqa: E402 - 延迟导入，UI 可独立加载
        role = _role_name(role)
    tag = f"{phase}·{role}" if phase and role else phase
    text = str(ev.get("text", ""))
    head, _, body = text.partition("\n")
    title = f"[{tag}] {head}" if tag else head
    return card(title, body.strip(), icon, state)


# ---------------- 侧边栏 / Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ 运行设置")
    mode = st.radio(
        "模式", ["offline（离线复现）", "live（书生 API）"], index=0,
        help="离线模式使用确定性脚本 LLM，评委可零配置复现；LIVE 模式调用 .env 中配置的书生兼容接口。",
    )
    mode_val = "offline" if mode.startswith("offline") else "live"
    max_steps = st.slider("最大步数 max_steps", 3, 12, 8)
    # P2-3：团队模式开关（默认关闭；单模型多角色编排，非多个独立智能体）
    team_mode = st.checkbox(
        "团队模式（检索员/计算员/评审员）", value=False,
        help="单模型多角色编排：同一基座按角色切换提示词与工具子集，步骤与账本打角色标签；不新增 LLM 调用。",
    )
    st.divider()
    st.markdown("### 🧪 快捷目标")
    presets = {
        "EGFR / VHH 预研（默认）": DEFAULT_GOAL,
        "pH 6.0 条件下的筛选一致性": "检索 pH 与亲和力测定一致性的要点，校验一条 VHH 候选序列，并写实验笔记。",
        "minibinder 多样性分桶": "检索 minibinder 长度与多样性分桶策略，校验一条 60 aa 候选，并写实验笔记。",
    }
    preset = st.selectbox("模板", list(presets))
    st.divider()
    demo_guard = st.button("🛡 反例演示：校验器拦截", use_container_width=True,
                           help="展示越界序列与违规笔记如何被工具层 / 校验器拦截")
    st.divider()
    st.caption("密钥仅从本地 `.env` 读取，界面与日志均不展示。")

os.environ["DEMO_MODE"] = mode_val

from xingzhi_agent.agent import ScienceAgent  # noqa: E402
from xingzhi_agent.config import load_settings  # noqa: E402

settings = load_settings()
effective_mode = "live" if settings.is_live else "offline"

_prev = st.session_state.get("result") or {}
state_phases = st.session_state.get("phases") or {
    e.get("phase") for e in _prev.get("events", []) if e.get("phase")
}
hero(effective_mode, settings.model if effective_mode == "live" else "OfflineLLM（确定性）", state_phases)

if mode_val == "live" and effective_mode == "offline":
    st.warning("未检测到可用的 `INTERNLM_API_KEY` / `INTERNLM_BASE_URL`，已自动回落到离线模式。请在 `.env` 中配置后重试。")

goal = st.text_area("🎯 科研目标（中文自然语言即可）", value=presets[preset], height=96)
run = st.button("▶  启动闭环", type="primary", use_container_width=True)

if demo_guard:
    from xingzhi_agent.tools import sequence_check  # noqa: E402
    from xingzhi_agent.verifier import verify_note_claims  # noqa: E402

    st.markdown("#### 🛡 反例演示：不可信输出会被拦截")
    c1, c2 = st.columns(2)
    with c1:
        bad_seq = sequence_check("ACDEFGHIKLZZ", "VHH")
        st.markdown("**① 越界 / 非法序列**")
        st.code("ACDEFGHIKLZZ", language="text")
        st.error(bad_seq.summary)
        st.json(bad_seq.data)
    with c2:
        # 反例演示用的占位符：sk-FAKE-KEY-DEMO 刻意写成不匹配密钥形态
        # （含连字符，校验器 SECRET_RE 要求 sk- 后跟 ≥12 位连续字母数字），
        # 仅用于演示“违规表述会被校验器拦截”，不含真实密钥；
        # 本条反例由禁用表述（官方指定 / 保证获奖 / 真实湿实验已测）触发拦截。
        bad_note = verify_note_claims(
            "本方案为官方指定方案，保证获奖，真实湿实验已测。key: sk-FAKE-KEY-DEMO",
            literature_ids=["P001"], sequence_ok=False,
        )
        st.markdown("**② 违规笔记（过度承诺）**")
        st.error(f"{bad_note['summary']}（{bad_note['score']}/100）")
        for i in bad_note["issues"]:
            st.markdown(f"❌ {i}")
    st.caption("这两条路径说明：智能体的输出必须通过工具层约束与校验器，才会被写入实验笔记。")
    st.divider()

kpi_slot = st.empty()
left, right = st.columns([1.05, 1], gap="large")
left.markdown("#### 🧭 控制环时间线")
right.markdown("#### 📓 实验笔记（Markdown 渲染）")
timeline_box = left.container()
note_box = right.container()

if run:
    st.session_state["phases"] = set()
    timeline_slot = timeline_box.empty()
    events: list[dict] = []
    # 运行中用 st.status 显示当前步骤，避免只剩一个静态 spinner、用户以为卡住；
    # 时间线只重绘最近 12 条（全量重绘是 O(n²)，长串卡片还会反复触发布局）。
    run_status = st.status("智能体推进中…", expanded=False)

    def on_event(ev: dict) -> None:
        events.append(ev)
        if ev.get("phase") in {p for p, _ in PHASES}:
            st.session_state["phases"].add(ev["phase"])
        step = ev.get("step")
        phase = ev.get("phase") or ""
        run_status.update(label=f"智能体推进中…（步骤 {step} · {phase}）" if step else f"智能体推进中…（{phase}）")
        shown = events[-12:]
        head = '<div style="color:var(--muted);font-size:.78rem;margin-bottom:6px">仅显示最近 12 条事件，完整时间线在运行结束后展示。</div>' if len(events) > 12 else ""
        timeline_slot.markdown(head + "".join(render_event(e) for e in shown), unsafe_allow_html=True)
        time.sleep(0.12)  # 轻微节奏感，便于录屏 / small pacing for screen recording

    with run_status:
        agent = ScienceAgent(settings, on_event=on_event, verbose=False, team=team_mode)
        result = agent.run(goal, max_steps=max_steps)

    st.session_state["result"] = {
        "mode": result.mode, "success": result.success, "error": result.error,
        "elapsed": result.elapsed_s, "note": result.note_markdown,
        "note_path": result.note_path, "verify": result.verify,
        "steps": result.steps, "events": events,
        "trajectory": result.score_trajectory, "repairs": result.repairs,
        "manifest": result.manifest, "literature_ids": result.literature_ids,
        "evidence": result.evidence_report,
        "hypotheses": result.hypothesis_report,
        "roles": result.roles,  # P2-3：团队模式时为角色汇总，否则为空 dict
    }
    st.rerun()  # 重绘一次：高亮已完成阶段 chips / repaint to light up phase chips

res = st.session_state.get("result")
if res:
    v = res.get("verify") or {}
    status = "通过" if res["success"] and v.get("passed") else ("完成（有提醒）" if res["success"] else "失败")
    cls = "good" if res["success"] and v.get("passed") else ("warn" if res["success"] else "bad")
    tools = [s.get("tool") for s in res["steps"] if s.get("tool")]
    kpi_slot.markdown(
        f"""
<div class="xz-kpi">
  <div class="k"><div class="lab">闭环状态</div><div class="val {cls}">{esc(status)}</div></div>
  <div class="k"><div class="lab">校验得分</div><div class="val {cls}">{esc(v.get('score', '—'))}/100</div></div>
  <div class="k"><div class="lab">工具调用</div><div class="val">{len(tools)}</div></div>
  <div class="k"><div class="lab">用时</div><div class="val">{res['elapsed']:.1f}s</div></div>
  <div class="k"><div class="lab">引用接地</div><div class="val {cls}">{esc(v.get('grounding_score', '—'))}%</div></div>
  <div class="k"><div class="lab">自修复轮次</div><div class="val">{res.get('repairs', 0)}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not run:
        timeline_box.markdown(
            "".join(render_event(e) for e in res.get("events", [])), unsafe_allow_html=True
        )

    if res.get("error"):
        st.error(f"运行未完成：{res['error']}")

    with note_box:
        if res.get("note"):
            body = re.sub(r"<!--.*?-->\s*", "", res["note"], flags=re.S).strip()
            # 把 [Pxxx] / [TOOL:x] 渲染成证据徽章，让"每句都有出处"一眼可见
            body = re.sub(r"^(#{1,2}) ", r"\1## ", body, flags=re.M)  # 笔记标题降级，避免压过页面标题
            body = re.sub(r"\[(P\d{3})\]", r"<span class='xz-cite'>\1</span>", body)
            body = re.sub(r"\[TOOL:([a-z_]+)\]", r"<span class='xz-cite tool'>\1</span>", body)
            with st.container(border=True):
                st.markdown(body, unsafe_allow_html=True)
            st.download_button(
                "⬇ 下载实验笔记 Markdown", res["note"],
                file_name=Path(res["note_path"] or "lab_note.md").name,
                mime="text/markdown", use_container_width=True,
            )
        else:
            st.info("本轮未产出笔记。可切回 offline 模式复现完整闭环。")

    with st.expander("🔍 校验器明细 / Verifier checks"):
        for c in v.get("checks", []):
            mark = "✅" if c.get("ok") else "❌"
            st.markdown(f"{mark} **{c['name']}** {('— ' + c['detail']) if c.get('detail') else ''}")
        if v.get("cited_ids"):
            st.caption("引用文献 ID：" + "、".join(v["cited_ids"]))

    traj = res.get("trajectory") or []
    if traj:
        st.markdown("#### 📈 自修复评分轨迹 / Critic-repair trajectory")
        render_trajectory(traj)

    if res.get("evidence"):
        st.markdown("#### ⚖ 证据辩证 / Evidence adjudication")
        evidence_panel(res["evidence"])

    if res.get("hypotheses"):
        st.markdown("#### 🏆 假设锦标赛 / Hypothesis tournament")
        hypothesis_panel(res["hypotheses"])

    # P2-3：团队分工面板（仅团队模式有数据；三角色的工具调用数均来自已有账本字段）
    if res.get("roles"):
        st.markdown("#### 👥 团队分工 / Team roles（单模型多角色编排）")
        for r in res["roles"].get("roles", []):
            st.markdown(
                f"- **{esc(r['name'])}**：{r['tool_calls']} 次工具调用"
                f"（{'、'.join(r['tools']) or '无'}），成功 {r['ok']} / 失败 {r['failed']} — {esc(r.get('description', ''))}"
            )
        st.caption(res["roles"].get("disclaimer", ""))

    seq_step = next((s_ for s_ in res["steps"] if s_.get("tool") == "sequence_check"), None)
    if seq_step and seq_step.get("tool_data", {}).get("metrics"):
        st.markdown("#### 🧪 理化指标 / Physicochemical metrics")
        mets = seq_step["tool_data"]["metrics"]
        metric_cards(mets)
        flags = seq_step["tool_data"].get("flags") or []
        for f in flags:
            st.warning(f)

    lit_step = next((s_ for s_ in res["steps"] if s_.get("tool") == "literature_search"), None)
    if lit_step:
        st.markdown("#### 📚 证据面板 / Evidence (Europe PMC, 可点击溯源)")
        for h in lit_step["tool_data"].get("hits", []):
            link = h.get("url") or ""
            meta = " · ".join(x for x in [str(h.get("year") or ""), h.get("venue") or "",
                                          ("DOI " + h["doi"]) if h.get("doi") else ""] if x)
            st.markdown(
                f'<div class="xz-card"><div class="t">[{esc(h["id"])}] '
                f'<a href="{esc(link)}" target="_blank" style="color:var(--accent);text-decoration:none">'
                f'{esc(h["title"])}</a></div>'
                f'<div class="d">{esc(meta)} · 相关度 {esc(h.get("score"))}<br>{esc((h.get("abstract") or "")[:240])}…</div></div>',
                unsafe_allow_html=True,
            )

    man = res.get("manifest") or {}
    st.markdown("#### 🧾 运行账本 / Run ledger")
    if man:
        run_ledger(man, "本次运行结果", (man.get("outputs") or {}).get("repair_stop_reason"))
        with st.expander("完整 manifest（inputs / outputs）"):
            st.code(json.dumps(
                {"inputs": man.get("inputs"), "outputs": {k: v_ for k, v_ in (man.get("outputs") or {}).items()
                                                          if k != "score_trajectory"},
                 "cost_cny_estimate": man.get("cost_cny_estimate"),
                 "reproducible": man.get("reproducible")},
                ensure_ascii=False, indent=2), language="json")
    else:
        _p, _m = newest_run(ROOT / "outputs" / "runs")
        if _m:
            run_ledger(_m, f"outputs/runs/{_p.name}")
        else:
            empty_state("暂无账本", "本次运行未产出 manifest。", "🧾")

    with st.expander("🛠 工具调用轨迹 / Tool traces（可逐步回放）"):
        from xingzhi_agent.roles import role_name as _role_name  # noqa: E402 - P2-3 角色中文名
        for s in res["steps"]:
            if not s.get("tool"):
                continue
            # P2-3：团队模式时步骤带角色标签
            role_tag = f" · {_role_name(s['role'])}" if s.get("role") else ""
            st.markdown(f"**步骤 {s['step']} · `{s['tool']}`{role_tag}** — {s.get('tool_summary','')}")
            st.code(json.dumps({"args": s.get("args"), "data": s.get("tool_data")},
                               ensure_ascii=False, indent=2)[:4000], language="json")
else:
    with timeline_box:
        empty_state("等待启动", "点击上方「启动闭环」，这里会逐步展示 意图 → 计划 → 工具 → 校验 → 写作 的全过程。", "🎬")
    with note_box:
        empty_state("尚无笔记", "闭环完成后，实验笔记会在此渲染，并可一键下载 Markdown。", "📄")
    st.markdown("#### 📈 自修复评分轨迹 / Critic-repair trajectory")
    _p, _m = newest_run(ROOT / "outputs" / "runs")
    _tr = ((_m.get("outputs") or {}).get("score_trajectory") or []) if _m else []
    if _tr:
        st.caption(f"尚未在本会话运行，展示最近一次存档：outputs/runs/{_p.name}")
        render_trajectory(_tr)
    else:
        empty_state("暂无评分轨迹", "运行一次闭环后，这里显示各修复轮次的质量分折线。", "📈")
    st.markdown("#### 🧪 理化指标 / Physicochemical metrics")
    empty_state("暂无序列指标", "智能体调用 sequence_check 后，长度、pI、GRAVY 等指标会以卡片展示。", "🧪")
    st.markdown("#### 🧾 运行账本 / Run ledger")
    if _m:
        run_ledger(_m, f"outputs/runs/{_p.name}（最近一次存档）")
    else:
        empty_state("暂无账本", "首次运行后会记录模型、逐步延迟、token、seed 与停止原因。", "🧾")

render_run_history()

st.markdown(
    '<div class="xz-foot">兴智杯 · 方向1 AI4S 科学智能体演示包 ｜ 离线可复现、证据可追溯、校验可审计。</div>',
    unsafe_allow_html=True,
)
