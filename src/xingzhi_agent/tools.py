"""工具层：文献检索 / 序列校验 / 写实验笔记。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .experiment import design_experiment as _design_experiment
from .figures import parse_figure as _parse_figure
from .physchem import analyze as _physchem

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.I)

VHH_LEN = (100, 150)
MINIBINDER_LEN = (40, 100)

TZ = ZoneInfo("Asia/Singapore")


@dataclass
class ToolResult:
    ok: bool
    name: str
    data: dict[str, Any]
    summary: str


def load_corpus(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("papers", []))


def _hit(paper: dict[str, Any], score: int) -> dict[str, Any]:
    """统一的命中结构，带真实来源字段（DOI / PMID / URL）。"""
    return {
        "id": paper.get("id"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "authors": paper.get("authors", [])[:4],
        "venue": paper.get("venue", ""),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", ""),
        "url": paper.get("url", ""),
        "score": score,
        "key_findings": paper.get("key_findings", []),
        "abstract": (paper.get("abstract", "") or "")[:420],
        # P2-1：图表元数据（label + 图注）随命中携带，供 [Pxxx:FigN] 接地流程使用；
        # 图注截断到 600 字，避免 LIVE 提示词被长图注撑爆（完整图注仍在 corpus.json 中）。
        "figures": [
            {"id": f.get("id", ""), "label": f.get("label", ""),
             "caption": (f.get("caption", "") or "")[:600],
             "graphic_uri": f.get("graphic_uri", "")}
            for f in (paper.get("figures") or [])
        ],
    }


# 中文查询术语桥（S1）：检索只做英文子串匹配，中文查询永远无命中；
# 先把中文词展开成英文同义词再检索，仍无命中才走默认推荐兜底。
_QUERY_BRIDGE: dict[str, list[str]] = {
    "纳米抗体": ["nanobody", "vhh"],
    "单域抗体": ["single-domain", "vhh", "sdab"],
    "抗体": ["antibody"],
    "亲和力": ["affinity"],
    "溶解度": ["solubility"],
    "稳定性": ["stability"],
    "表达": ["expression"],
    "筛选": ["screening"],
    "膜蛋白": ["membrane"],
    "靶点": ["target"],
    "结构": ["structure"],
    "预测": ["prediction"],
    "设计": ["design"],
    "纯化": ["purification"],
    "结合": ["binding"],
    "文库": ["library"],
    "突变": ["mutation"],
    "聚集": ["aggregation"],
    "产量": ["yield"],
}


def literature_search(
    corpus: list[dict[str, Any]],
    query: str,
    top_k: int = 4,
) -> ToolResult:
    """字段加权 + IDF 的本地检索 / field-weighted BM25-lite retrieval over the local corpus.

    标题 ×3、关键词 ×2、摘要 ×1；稀有词按 IDF 加权，避免 "protein" 这类高频词主导排序。
    无精确命中时返回按年份默认推荐，并打显性 fallback 标记（S1），不得无声降级。
    """
    top_k = max(1, int(top_k))  # O4：top_k=0/-1 时钳制为 1，避免空命中 IndexError 与 [:-1] 切片越界
    tokens = [t.lower() for t in re.split(r"[\s,，、;；/]+", query.strip()) if len(t) > 1]
    if not tokens:
        tokens = ["vhh"]
    # 中文查询先经术语桥展开成英文同义词（S1），再参与英文子串匹配
    expanded: list[str] = list(tokens)
    for t in tokens:
        for cn, ens in _QUERY_BRIDGE.items():
            if cn in t:
                expanded.extend(ens)
    tokens = expanded

    fields: list[tuple[dict[str, Any], str, str, str]] = []
    for paper in corpus:
        title = (paper.get("title") or "").lower()
        kw = " ".join(paper.get("keywords", [])).lower()
        body = ((paper.get("abstract") or "") + " " + " ".join(paper.get("key_findings", []))).lower()
        fields.append((paper, title, kw, body))

    n = max(1, len(fields))
    df = {
        t: sum(1 for _, ti, kw, bo in fields if t in ti or t in kw or t in bo)
        for t in set(tokens)
    }
    idf = {t: math.log((n + 1) / (df.get(t, 0) + 1)) + 0.2 for t in set(tokens)}

    scored: list[tuple[float, dict[str, Any]]] = []
    for paper, title, kw, body in fields:
        score = 0.0
        for t in tokens:
            hits = 3 * title.count(t) + 2 * kw.count(t) + body.count(t)
            if hits:
                score += idf[t] * (1 + math.log(hits))
        if score > 0:
            # 轻微偏好近 5 年与高被引（真实语料里这两项都有值）
            recency = max(0, int(paper.get("year", 0)) - 2018) * 0.05
            cited = math.log1p(int(paper.get("citations", 0))) * 0.08
            scored.append((score + recency + cited, paper))

    scored.sort(key=lambda x: (-x[0], -int(x[1].get("year", 0))))
    hits = [_hit(p, round(sc, 3)) for sc, p in scored[:top_k]]
    fallback = False
    if not hits:
        if not corpus:
            return ToolResult(ok=False, name="literature_search",
                              data={"hits": [], "query": query, "fallback": False},
                              summary="文献检索无结果：语料为空。")
        for paper in sorted(corpus, key=lambda p: -int(p.get("year", 0)))[:top_k]:
            h = _hit(paper, 0)
            h["note"] = "无精确命中，返回默认推荐"
            h["fallback"] = True  # S1：兜底命中打显性标记
            hits.append(h)
        fallback = True
    summary = (
        f"文献检索完成：query={query!r}，命中 {len(hits)} 篇"
        f"（top: {hits[0]['id']} {hits[0]['title'][:40]}…）"
    )
    if fallback:
        summary += "（无精确命中，按年份默认推荐，不可作为检索证据引用）"
    return ToolResult(ok=True, name="literature_search",
                      data={"hits": hits, "query": query, "fallback": fallback}, summary=summary)


def sequence_check(
    sequence: str,
    molecule_type: str = "VHH",
    assay_ph: float = 6.0,
) -> ToolResult:
    """序列约束 + 理化指标校验 / sequence constraints plus physicochemical metrics."""
    # O4：assay_ph 必须在 [0, 14] 内，否则明确报错
    if not isinstance(assay_ph, (int, float)) or isinstance(assay_ph, bool) or not (0 <= assay_ph <= 14):
        return ToolResult(
            ok=False, name="sequence_check",
            data={"molecule_type": molecule_type, "assay_ph": assay_ph, "ok": False,
                  "issues": [f"assay_ph={assay_ph} 超出范围 [0, 14]"], "flags": [], "metrics": []},
            summary=f"序列校验失败：assay_ph={assay_ph} 超出范围 [0, 14]",
        )
    rep = _physchem(sequence, molecule_type, assay_ph).as_dict()
    # S10：未知类型由 physchem.analyze 明确报错；此处不再把未知类型静默映射为 VHH
    if rep["issues"] and any("未知 molecule_type" in i for i in rep["issues"]):
        return ToolResult(
            ok=False, name="sequence_check",
            data={"molecule_type": molecule_type, "assay_ph": assay_ph, "length": len(re.sub(r"\s+", "", sequence or "")),
                  "ok": False, "issues": rep["issues"], "flags": [], "metrics": [],
                  "sequence_preview": "", "sequence_sha256": ""},
            summary="序列校验失败：" + "；".join(rep["issues"]),
        )
    seq = re.sub(r"\s+", "", sequence or "").upper()
    mtype = "minibinder" if molecule_type.strip().upper() in {"MINIBINDER", "MINI", "MB"} else "VHH"
    data = {
        "molecule_type": mtype,
        "assay_ph": assay_ph,
        "length": len(seq),
        "length_rule": list(MINIBINDER_LEN if mtype == "minibinder" else VHH_LEN),
        "ok": rep["ok"],
        "issues": rep["issues"],
        "flags": rep["flags"],
        "metrics": rep["metrics"],
        "sequence_preview": (seq[:20] + "…") if len(seq) > 20 else seq,
        "sequence_sha256": __import__("hashlib").sha256(seq.encode()).hexdigest()[:16],
    }
    failed = [m["name"] for m in rep["metrics"] if m["ok"] is False]
    if rep["ok"]:
        summary = (
            f"理化校验通过：{mtype}，长度 {data['length']}，"
            f"pI={_metric(rep, '等电点 pI')}，GRAVY={_metric(rep, 'GRAVY')}，"
            f"pH{assay_ph} 净电荷={_metric(rep, f'净电荷 @pH {assay_ph}')}"
            + (f"；提醒 {len(rep['flags'])} 条" if rep["flags"] else "")
        )
    else:
        summary = "序列校验失败：" + "；".join(rep["issues"] or failed)
    return ToolResult(ok=rep["ok"], name="sequence_check", data=data, summary=summary)


def _metric(rep: dict[str, Any], name: str) -> Any:
    for m in rep["metrics"]:
        if m["name"] == name:
            return m["value"]
    return "—"


def design_experiment(
    sequence: str,
    molecule_type: str = "VHH",
    assay_ph: float = 6.0,
    target: str = "EGFR",
) -> ToolResult:
    """下一步实验设计建议 / propose the next discriminative experiment."""
    # O4：assay_ph 必须在 [0, 14] 内，否则明确报错
    if not isinstance(assay_ph, (int, float)) or isinstance(assay_ph, bool) or not (0 <= assay_ph <= 14):
        return ToolResult(
            ok=False, name="design_experiment",
            data={"target": target, "assay_ph": assay_ph, "physchem_ok": False,
                  "error": f"assay_ph={assay_ph} 超出范围 [0, 14]", "ranked_experiments": []},
            summary=f"实验设计失败：assay_ph={assay_ph} 超出范围 [0, 14]",
        )
    plan = _design_experiment(sequence, molecule_type, assay_ph, target)
    if plan.get("error") or not plan.get("ranked_experiments"):
        # S10：未知类型等理化失败时，实验设计器已返回明确错误，此处直接透传为工具失败
        return ToolResult(
            ok=False, name="design_experiment", data=plan,
            summary=f"实验设计失败：{plan.get('error') or '无可用实验方案'}",
        )
    top = plan["ranked_experiments"][0]
    return ToolResult(
        ok=True,
        name="design_experiment",
        data=plan,
        summary=f"实验设计建议：优先 {top['experiment'][:48]}… （判别价值：{top['discriminative_value'][:28]}…）",
    )


def write_note(
    outputs_dir: Path,
    title: str,
    body_markdown: str,
) -> ToolResult:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", title.strip())[:80] or "lab_note"
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    path = outputs_dir / f"{safe}_{ts}.md"
    header = (
        f"<!-- generated_at: {datetime.now(TZ).isoformat()} Asia/Singapore -->\n"
        f"<!-- title: {title} -->\n\n"
    )
    path.write_text(header + body_markdown.strip() + "\n", encoding="utf-8")
    # 同步一份稳定文件名，方便 README / 提交包引用
    latest = outputs_dir / "latest_lab_note.md"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    summary = f"实验笔记已写入：{path}"
    return ToolResult(
        ok=True,
        name="write_note",
        data={"path": str(path), "latest": str(latest), "title": title},
        summary=summary,
    )



def _coerce_float(value: Any, default: float) -> float:
    """宽松解析数值：接受 5.0 / "5.0" / "pH5.0" / "pH 7.4" / "7.4 (25C)" 等。

    O4：范围写法（如 "5.0-6.0"）直接拒绝，不再静默取第一个数——
    调用方 run_tool 会把它收敛为可观测的工具失败，交回模型修正参数后重试。
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return float(default)
    text = str(value)
    if re.search(r"-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?", text):
        raise ValueError(f"数值参数 {value!r} 为范围写法，请只给单个数字")
    import re as _re
    m = _re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else float(default)


def _coerce_int(value: Any, default: int) -> int:
    return int(_coerce_float(value, float(default)))


def _run_tool_inner(
    name: str,
    args: dict[str, Any],
    *,
    corpus: list[dict[str, Any]],
    outputs_dir: Path,
) -> ToolResult:
    name = name.strip()
    if name == "literature_search":
        return literature_search(
            corpus,
            query=str(args.get("query", "EGFR VHH")),
            top_k=_coerce_int(args.get("top_k", 4), 4),
        )
    if name == "sequence_check":
        return sequence_check(
            sequence=str(args.get("sequence", "")),
            molecule_type=str(args.get("molecule_type", "VHH")),
            assay_ph=_coerce_float(args.get("assay_ph", 6.0), 6.0),
        )
    if name == "design_experiment":
        return design_experiment(
            sequence=str(args.get("sequence", "")),
            molecule_type=str(args.get("molecule_type", "VHH")),
            assay_ph=_coerce_float(args.get("assay_ph", 6.0), 6.0),
            target=str(args.get("target", "EGFR")),
        )
    if name == "parse_figure":
        # P2-1：图表解析——对已 baked 进语料的图注做规则数值抽取，不联网、不读图像素
        res = _parse_figure(
            corpus,
            paper_id=str(args.get("paper_id", args.get("paper", ""))),
            fig_ref=str(args.get("fig_id", args.get("fig_ref", "Fig1"))),
        )
        ok = bool(res.get("ok"))
        nums = res.get("numbers", []) or []
        summary = (
            f"图表解析：{res.get('paper_id')} {res.get('label', '')}，"
            f"抽到 {len(nums)} 个数值" + (f"（如 {nums[0]['raw']}）" if nums else "")
            if ok else f"图表解析失败：{res.get('error', '未知错误')}"
        )
        return ToolResult(ok=ok, name="parse_figure", data=res, summary=summary)
    if name == "write_note":
        return write_note(
            outputs_dir,
            title=str(args.get("title", "实验笔记")),
            body_markdown=str(args.get("body_markdown", args.get("body", ""))),
        )
    return ToolResult(
        ok=False,
        name=name,
        data={},
        summary=f"未知工具：{name}",
    )


def run_tool(
    name: str,
    args: dict[str, Any],
    *,
    corpus: list[dict[str, Any]],
    outputs_dir: Path,
) -> ToolResult:
    """工具调用永不抛出：参数异常时降级为可观测的失败结果，交回模型重试。"""
    try:
        return _run_tool_inner(name, args, corpus=corpus, outputs_dir=outputs_dir)
    except Exception as exc:  # noqa: BLE001 - 故意兜底，保证控制环不崩
        return ToolResult(
            ok=False,
            name=name,
            data={"error": type(exc).__name__, "args": {k: str(v)[:120] for k, v in (args or {}).items()}},
            summary=f"工具 {name} 调用失败：{type(exc).__name__}: {exc}。请修正参数后重试（数值参数请只给数字）。",
        )
