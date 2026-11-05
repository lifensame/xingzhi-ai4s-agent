"""Dashboard panels: run ledger, metric cards, empty states."""
from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

STOP_LABELS = {
    "target_reached": "达到 90 分目标", "rounds_exhausted": "修复轮次用尽",
    "budget_exhausted": "时间预算用尽", "call_failure": "调用失败停止",
    "repair_disabled": "未启用修复", "": "—",
}


def esc(x) -> str:
    return html.escape(str(x))


def newest_run(runs_dir: Path) -> tuple[Path | None, dict]:
    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            return p, json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return None, {}


def empty_state(title: str, detail: str, icon: str = "◌") -> None:
    st.markdown(
        f'<div class="xz-empty"><div class="i">{esc(icon)}</div>'
        f'<div class="t">{esc(title)}</div><div class="d">{esc(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, css_class)."""
    cells = "".join(
        f'<div class="k"><div class="lab">{esc(l)}</div><div class="val {c}">{esc(v)}</div></div>'
        for l, v, c in items
    )
    st.markdown(f'<div class="xz-kpi">{cells}</div>', unsafe_allow_html=True)


def metric_cards(mets: list[dict]) -> None:
    cells = []
    for m in mets[:10]:
        ok = m.get("ok")
        cls = "good" if ok else ("bad" if ok is False else "")
        mark = "✓" if ok else ("✗" if ok is False else "·")
        cells.append(
            f'<div class="xz-metric"><div class="lab">{esc(m.get("name", ""))}</div>'
            f'<div class="val {cls}">{esc(m.get("value", "—"))}<span class="u">{esc(m.get("unit", ""))}</span></div>'
            f'<div class="th"><span class="{cls}">{mark}</span> 阈值 {esc(m.get("threshold") or "—")}</div></div>'
        )
    st.markdown(f'<div class="xz-metrics">{"".join(cells)}</div>', unsafe_allow_html=True)


def run_ledger(man: dict, source: str, stop_reason: str | None = None) -> None:
    """Model, per-step latency, token counts, seed and stop reason."""
    inputs = man.get("inputs") or {}
    outs = man.get("outputs") or {}
    stop = stop_reason if stop_reason is not None else outs.get("repair_stop_reason", "")
    if not stop and (outs.get("verify_score") or 0) >= (inputs.get("repair_target") or 90):
        stop = "target_reached"
    lat = man.get("latency_s") or {}
    kpi_row([
        ("模型", man.get("model", "—"), ""),
        ("seed", inputs.get("seed", "—"), ""),
        ("停止原因", STOP_LABELS.get(stop or "", stop or "—"), "good" if stop == "target_reached" else ""),
        ("总 token", f'{man.get("total_tokens", 0):,}', ""),
        ("输入 / 输出 token", f'{man.get("prompt_tokens", 0):,} / {man.get("completion_tokens", 0):,}', ""),
        ("总墙钟", f'{lat.get("total_wall", 0)}s', ""),
    ])
    steps = man.get("steps") or []
    if steps:
        rows = [{"步": s.get("step"), "阶段": "修复" if s.get("phase") == "repair" else "主环",
                 "模型": s.get("model", ""), "状态": "✓" if s.get("ok", True) else "✗ 失败",
                 "延迟 s": s.get("latency_s", 0), "prompt tok": s.get("prompt_tokens", 0),
                 "completion tok": s.get("completion_tokens", 0),
                 "prompt sha256": (s.get("prompt_sha256") or "")[:12]} for s in steps]
        st.dataframe(rows, hide_index=True, use_container_width=True,
                     column_config={"延迟 s": st.column_config.ProgressColumn(
                         "延迟 s", format="%.2f", min_value=0,
                         max_value=max(float(r["延迟 s"] or 0) for r in rows) or 1.0)})
    st.caption(f"来源：{source} · temperature {inputs.get('temperature', '—')} · "
               f"{inputs.get('seed_policy', '')}")


LEVEL_STYLE = {
    "direct": ("直接支持", "good"), "indirect": ("间接支持", ""),
    "overstated": ("过度表述", "warn"), "unsupported": ("无支撑", "bad"),
}
KIND_STYLE = {
    "real": ("实质分歧", "bad"), "conditions": ("条件差异（伪冲突）", "warn"),
    "definitions": ("口径差异（伪冲突）", "warn"),
}


def evidence_panel(report: dict) -> None:
    """证据辩证面板：逐句支撑强度、过度表述降级、文献分歧与置信度。"""
    if not report or report.get("error"):
        empty_state("尚无证据辩证结果", report.get("error") if report else "运行一次后在此查看逐句支撑强度与文献分歧", "⚖")
        return
    sup = report.get("support") or {}
    cnt = sup.get("counts") or {}
    dcnt = (report.get("disagreements") or {}).get("counts") or {}
    kpi_row([
        ("支撑分", f'{sup.get("support_score", "—")}/100', "good"),
        ("直接支持", str(cnt.get("direct", 0)), "good"),
        ("间接支持", str(cnt.get("indirect", 0)), ""),
        ("过度表述（已降级）", str(cnt.get("overstated", 0)), "warn"),
        ("无支撑", str(cnt.get("unsupported", 0)), "bad" if cnt.get("unsupported") else ""),
        ("实质分歧", str(dcnt.get("real", 0)), "bad" if dcnt.get("real") else ""),
    ])

    rows = sup.get("rows") or []
    if rows:
        st.dataframe(
            [{"支撑强度": LEVEL_STYLE.get(r["level"], (r["level"], ""))[0],
              "重叠": r.get("overlap", 0), "证据": r.get("matched_id") or "—",
              "句子": r.get("text", ""), "判定理由": r.get("reason", "")} for r in rows],
            hide_index=True, use_container_width=True,
            column_config={"重叠": st.column_config.ProgressColumn("重叠", format="%.2f", min_value=0, max_value=1)},
        )
    softened = sup.get("softened") or []
    if softened:
        st.markdown("**自动降级的过度表述**")
        for s in softened:
            st.markdown(
                f'<div class="xz-card"><div class="t">改写前：{esc(s["before"])}</div>'
                f'<div class="d" style="color:var(--accent)">改写后：{esc(s["after"])}</div></div>',
                unsafe_allow_html=True,
            )
    pairs = (report.get("disagreements") or {}).get("disagreements") or []
    if pairs:
        st.markdown("**文献分歧检测**（区分实质分歧与条件 / 口径差异导致的伪冲突）")
        st.dataframe(
            [{"类型": KIND_STYLE.get(p["kind"], (p["kind"], ""))[0], "主题": p["topic"],
              "文献 A": p["a"], "文献 B": p["b"], "判定依据": p["reason"]} for p in pairs],
            hide_index=True, use_container_width=True,
        )
    confs = report.get("confidence") or []
    if confs:
        st.markdown("**各主题结论置信度**（来源数量 × 方法多样性 × 分歧情况）")
        st.dataframe(
            [{"主题": c["topic"], "置信度": c["confidence"], "等级": c["level"],
              "来源数": c["sources"], "方法种类": c["method_variety"], "实质分歧": c["real_conflicts"]}
             for c in confs],
            hide_index=True, use_container_width=True,
            column_config={"置信度": st.column_config.ProgressColumn("置信度", format="%d", min_value=0, max_value=100)},
        )


ORIGIN_CN = {"conflict": "源自文献分歧", "gap": "源自证据缺口"}


def hypothesis_panel(report: dict) -> None:
    """假设锦标赛面板：Elo 排行、支持/反证计数、存活分与下一步判别实验。"""
    if not report or report.get("error"):
        empty_state("尚无假设锦标赛结果", report.get("error") if report else "运行一次后在此查看假设排名", "🏆")
        return
    hyps = report.get("hypotheses") or []
    if not hyps:
        empty_state("本轮未生成假设", "语料中未检出可供生成假设的分歧或缺口", "🏆")
        return
    top = hyps[0]
    kpi_row([
        ("假设数", str(len(hyps)), ""),
        ("对决场次", str(report.get("rounds", 0)), ""),
        ("冠军假设", top["id"], "good"),
        ("冠军 Elo", str(top["elo"]), "good"),
        ("冠军存活分", f'{top["survival"]}/100', "good" if top["survival"] >= 60 else "warn"),
    ])
    DIR_CN = {"up": "升", "down": "降", "neutral": "中性"}
    st.dataframe(
        [{"排名": i + 1, "假设": h["id"], "Elo": h["elo"], "存活分": h["survival"],
          "对抗": h.get("contest", 0),
          "胜/负": f'{h["wins"]}/{h["losses"]}',
          "支持": "、".join(h["support"][:3]) or "—",
          "反证": "、".join(h["counter"][:3]) or "—",
          "主张方向": DIR_CN.get(h.get("claim_direction", ""), h.get("claim_direction", "")),
          "来源": ORIGIN_CN.get(h["origin"], h["origin"]), "主题": h["topic"], "陈述": h["text"]}
         for i, h in enumerate(hyps)],
        hide_index=True, use_container_width=True,
        column_config={
            "Elo": st.column_config.ProgressColumn(
                "Elo", format="%.0f",
                min_value=min(h["elo"] for h in hyps) - 20, max_value=max(h["elo"] for h in hyps) + 20),
            "存活分": st.column_config.ProgressColumn("存活分", format="%d", min_value=0, max_value=100),
        },
    )
    nx = report.get("next_experiment") or {}
    if nx:
        st.markdown(
            f'<div class="xz-card"><div class="t">下一步最小判别实验（针对 {esc(nx.get("hypothesis", ""))}）</div>'
            f'<div class="d">{esc(nx.get("design", ""))}</div>'
            f'<div class="d">读出：{esc(nx.get("readout", ""))}</div>'
            f'<div class="d" style="color:var(--accent)">判据：{esc(nx.get("decision", ""))}</div></div>',
            unsafe_allow_html=True,
        )
