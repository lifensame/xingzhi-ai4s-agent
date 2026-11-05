"""多模态（图表）证据：图注解析与数值抽取 / Figure-caption parsing (P2-1).

最小可信实现，只做“图注元数据解析”：
- 图表元数据来自 Europe PMC fullTextXML 的 <fig>（图注 caption + 图形链接），
  由 scripts/fetch_corpus.py 抓取并 baked 进 data/corpus.json 的 `figures` 字段；
  离线运行时不联网，只读已 baked 的字段。
- 本模块对图注做规则数值抽取（数值 + 单位 + 条件），不做图像像素 OCR、
  不做视觉模型读图；无图注的论文如实返回“无图表元数据”，不编造。
- 图表主张引用格式为 [Pxxx:FigN]（FigN 按该论文 figures 列表顺序从 1 编号），
  接入 grounding.py / evidence.py 的现有接地流程，图注即证据文本。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable

# 图表引用：[P007:Fig1]；FigN 按该论文 figures 列表顺序从 1 编号
FIG_CITE_RE = re.compile(r"\[(P\d{3}):Fig(\d+)\]")

# 数值 + 单位：覆盖图注常见写法（浓度、温度、pH、分子量、倍数、百分比、时间、尺寸）
_NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT = (
    r"nM|µM|μM|uM|mM|M|pM|mg/mL|µg/mL|ug/mL|ng/mL|kDa|Da|°C|℃|%|fold|×|x|"
    r"nm|µm|μm|mm|cm|h|min|s|aa|bp|mL|µL|uL|nM/L|molar"
)
NUMBER_RE = re.compile(rf"({_NUM})\s*-?\s*({_UNIT})\b")  # 允许 "20-fold" 连字符写法
# 条件：pH、温度、浓度、时间点（如 pH 6.0 / at 25 °C / 24 h）
COND_RE = re.compile(
    r"(pH\s*[+-]?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:°C|℃)|"
    r"\d+(?:\.\d+)?\s*(?:nM|µM|μM|uM|mM|mg/mL|µg/mL|ug/mL)|"
    r"(?:at|after|for)\s+\d+(?:\.\d+)?\s*(?:h|min|s))",
    re.I,
)
# 纯数字（无单位）：如 "n = 3"、"EC50 12.5"、"10-fold" 前的倍数已由 NUMBER_RE 覆盖
BARE_NUM_RE = re.compile(rf"\b(?:n|EC50|IC50|KD|Kd|kD|Tm|t1/2|R2)\s*[=:≈]?\s*({_NUM})\b")


@dataclass
class FigureNumber:
    value: float
    unit: str
    raw: str
    context: str = ""  # 数值前后约 40 字，供人工复核

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "raw": self.raw, "context": self.context}


def extract_numbers(caption: str) -> list[FigureNumber]:
    """对图注做规则数值抽取：数值 + 单位 + 上下文；另抽条件（pH/温度/浓度）。

    只做确定性正则抽取，不推断、不补全；同一数值不在不同写法下重复计入。
    """
    out: list[FigureNumber] = []
    seen: set[tuple[float, str, int]] = set()
    text = caption or ""
    for m in NUMBER_RE.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        key = (val, m.group(2), m.start())
        if key in seen:
            continue
        seen.add(key)
        ctx = text[max(0, m.start() - 40):m.end() + 40].strip()
        out.append(FigureNumber(value=val, unit=m.group(2), raw=m.group(0), context=ctx))
    for m in BARE_NUM_RE.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        key = (val, "", m.start())
        if key in seen:
            continue
        seen.add(key)
        ctx = text[max(0, m.start() - 40):m.end() + 40].strip()
        out.append(FigureNumber(value=val, unit="", raw=m.group(0), context=ctx))
    return out


def extract_conditions(caption: str) -> list[str]:
    """抽取图注中的实验条件（pH/温度/浓度/时长），去重保序。"""
    seen: set[str] = set()
    out: list[str] = []
    for m in COND_RE.finditer(caption or ""):
        c = re.sub(r"\s+", " ", m.group(0)).strip()
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def find_figure(paper: dict[str, Any] | None, fig_ref: str) -> dict[str, Any] | None:
    """按引用解析图表：FigN（按 figures 列表顺序从 1 编号）、FIGURE N 标签、原始 fig id 均可。

    找不到时返回 None（调用方据此报“无图表元数据”，不得编造）。
    """
    if not paper:
        return None
    figs = paper.get("figures") or []
    if not figs:
        return None
    ref = (fig_ref or "").strip()
    m = re.fullmatch(r"Fig(\d+)", ref, re.I)
    if m:
        idx = int(m.group(1)) - 1
        return dict(figs[idx]) if 0 <= idx < len(figs) else None
    m = re.fullmatch(r"(?:FIGURE|Figure|Fig\.?)\s*(\d+)", ref, re.I)
    if m:
        idx = int(m.group(1)) - 1
        return dict(figs[idx]) if 0 <= idx < len(figs) else None
    for f in figs:
        if str(f.get("id", "")) == ref or str(f.get("label", "")).strip().lower() == ref.lower():
            return dict(f)
    return None


def parse_figure(
    corpus: Iterable[dict[str, Any]],
    paper_id: str,
    fig_ref: str = "Fig1",
) -> dict[str, Any]:
    """解析指定论文的指定图表：返回图注 + 抽取的数值与条件。

    返回 dict（tools.parse_figure 工具会包成 ToolResult）：
    ok=True 时含 paper_id / fig_id / label / caption / graphic_uri / numbers / conditions；
    ok=False 时含 error（论文不存在 / 该论文无图表元数据 / 图表编号不存在），不编造。
    """
    papers = {str(p.get("id")): p for p in corpus if p.get("id")}
    paper = papers.get(str(paper_id))
    if paper is None:
        return {"ok": False, "paper_id": paper_id, "fig_ref": fig_ref,
                "error": f"语料中无此论文：{paper_id}"}
    figs = paper.get("figures")
    if not figs:
        return {"ok": False, "paper_id": paper_id, "fig_ref": fig_ref,
                "error": f"{paper_id} 无图表元数据（未抓到开放获取全文，或全文无 <fig> 图注）"}
    fig = find_figure(paper, fig_ref)
    if fig is None:
        return {"ok": False, "paper_id": paper_id, "fig_ref": fig_ref,
                "error": f"{paper_id} 无此图表：{fig_ref}（该论文共 {len(figs)} 张图）"}
    caption = str(fig.get("caption") or "")
    numbers = extract_numbers(caption)
    return {
        "ok": True,
        "paper_id": paper_id,
        "fig_ref": fig_ref,
        "fig_id": fig.get("id", ""),
        "label": fig.get("label", ""),
        "caption": caption,
        "graphic_uri": fig.get("graphic_uri", ""),
        "numbers": [n.as_dict() for n in numbers],
        "conditions": extract_conditions(caption),
    }


def figure_caption(evidence: Iterable[dict[str, Any]], paper_id: str, fig_no: int) -> str | None:
    """从本轮证据条目中取指定图表的图注文本（供接地流程用）；取不到返回 None。"""
    for e in evidence:
        if str(e.get("id")) != str(paper_id):
            continue
        figs = e.get("figures") or []
        if 1 <= fig_no <= len(figs):
            return str((figs[fig_no - 1] or {}).get("caption") or "") or None
        return None
    return None


def figures_summary(corpus: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """语料图表覆盖统计（供 UI / README / 账本如实展示，不编造）。"""
    papers = list(corpus)
    with_fig = [p for p in papers if p.get("figures")]
    no_fig = [p for p in papers if not p.get("figures")]
    total = sum(len(p.get("figures") or []) for p in papers)
    return {
        "papers_total": len(papers),
        "papers_with_figures": len(with_fig),
        "papers_without_figures": len(no_fig),
        "figures_total": total,
        "papers_with_figures_ids": [str(p.get("id")) for p in with_fig],
    }


# ---------- fullTextXML → figures（供 scripts/fetch_corpus.py 使用） ----------

def _fig_from_pyeuropepmc(xml_text: str) -> list[dict[str, str]] | None:
    """优先用 PyEuropePMC（MIT）的 FullTextXMLParser 抽 <fig>；不可用时返回 None 走兜底。"""
    try:
        from pyeuropepmc import FullTextXMLParser
    except Exception:
        return None
    try:
        figs = FullTextXMLParser(xml_text).extract_figures()
    except Exception:
        return None
    out: list[dict[str, str]] = []
    for f in figs or []:
        out.append({
            "id": str(f.get("id", "")),
            "label": str(f.get("label", "")),
            "caption": str(f.get("caption", "")),
            "graphic_uri": str(f.get("graphic_uri", "")),
        })
    return out


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _fig_from_elementtree(xml_text: str) -> list[dict[str, str]]:
    """stdlib 兜底：直接解析 JATS <fig>（label + caption 文本 + graphic 链接），不依赖第三方库。"""
    root = ET.fromstring(xml_text)
    out: list[dict[str, str]] = []
    for fig in root.iter():
        if _localname(fig.tag) != "fig":
            continue
        label, caption, graphic = "", "", ""
        for child in fig.iter():
            name = _localname(child.tag)
            if name == "label" and not label:
                label = "".join(child.itertext()).strip()
            elif name == "caption" and not caption:
                caption = re.sub(r"\s+", " ", "".join(child.itertext())).strip()
            elif name == "graphic" and not graphic:
                for k, v in child.attrib.items():
                    if _localname(k) == "href":
                        graphic = v
        out.append({
            "id": str(fig.attrib.get("id", "")),
            "label": label,
            "caption": caption,
            "graphic_uri": graphic,
        })
    return out


def extract_figures_from_xml(xml_text: str) -> list[dict[str, str]]:
    """从 Europe PMC fullTextXML 抽取图表（label + caption + graphic_uri）。

    优先 PyEuropePMC，失败/未安装时用标准库 ElementTree 兜底；两者都失败返回 []。
    图注为空的 <fig> 会被保留（caption 为 ""），由调用方如实记录，不编造。
    """
    figs = _fig_from_pyeuropepmc(xml_text)
    if figs is not None:
        return figs
    try:
        return _fig_from_elementtree(xml_text)
    except Exception:
        return []
