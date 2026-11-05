"""补丁式自修复 + 引用白名单 / patch-shaped repair and citation whitelist.

- evidence_pack：把本轮命中文献的片段内联给模型，并给出可引用 ID 白名单；
- sanitize_citations：确定性后处理，删除白名单外的 ID；只剩未知 ID 的句子标记为未接地并移除；
- failed_sections / splice_sections：只让模型重写未通过的小节，再按标题确定性拼回。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

ID_RE = re.compile(r"P\d{3}")
BRACKET_RE = re.compile(r"\[([^\[\]]*P\d{3}[^\[\]]*)\]")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def evidence_pack(hits: Iterable[dict[str, Any]], max_abs: int = 260) -> str:
    lines = []
    for h in hits:
        if not h.get("id"):
            continue
        kf = "；".join(h.get("key_findings", [])[:3])
        abs_ = (h.get("abstract") or "")[:max_abs]
        lines.append(f"[{h['id']}] {h.get('title','')} ({h.get('year','')})\n  要点：{kf}\n  摘要：{abs_}")
    return "\n".join(lines)


def whitelist_rules(ids: list[str]) -> str:
    return (
        f"【引用白名单】只能引用：{', '.join(ids)}。\n"
        "规则：每个事实句只写一个结论，并且句末恰好带一个 ID（如 [P003]）；每个含数字、比较、因果或建议的句子，必须在句末以 [Pxxx] 结尾；"
        "所引文献的『要点/摘要』中必须出现该句的关键术语（尽量沿用原文英文术语，如 nanobody、VHH、pH、aggregation）。"
        "工具算出的数值句末用 [TOOL:sequence_check] 或 [TOOL:design_experiment]。"
        "找不到支撑的句子请删掉，不要写无引用的事实句。不要使用白名单以外的 ID。"
    )


def sanitize_citations(body: str, whitelist: Iterable[str]) -> tuple[str, dict[str, Any]]:
    wl = set(whitelist)
    stats = {"stripped_ids": [], "dropped_lines": []}
    out = []
    for line in body.splitlines():
        ids = ID_RE.findall(" ".join(BRACKET_RE.findall(line)))
        if not ids or not wl:
            out.append(line); continue
        bad = [i for i in ids if i not in wl]
        if not bad:
            out.append(line); continue
        stats["stripped_ids"].extend(bad)
        if all(i in bad for i in ids) and "TOOL:" not in line:
            stats["dropped_lines"].append(line.strip()[:120])
            continue

        def fix(m: re.Match) -> str:
            keep = [i for i in ID_RE.findall(m.group(1)) if i in wl]
            return f"[{', '.join(keep)}]" if keep else ""
        out.append(BRACKET_RE.sub(fix, line))
    return "\n".join(out), stats


def split_sections(body: str) -> list[tuple[str, str]]:
    """[(heading or '', text)]，heading 为 '## xxx' 行。"""
    parts: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        if H2_RE.match(line):
            parts.append((line.strip(), []))
        else:
            parts[-1][1].append(line)
    return [(h, "\n".join(ls)) for h, ls in parts if h or "".join(ls).strip()]


def _key(heading: str) -> str:
    t = H2_RE.match(heading).group(1) if H2_RE.match(heading) else heading
    return re.sub(r"^[\d.、\s一二三四五六七八九十]+", "", t).strip()


def failed_sections(body: str, verify: dict[str, Any], required: Iterable[str]) -> list[str]:
    """返回需重写的小节标题（含缺失小节名）。前言区（第一个 ## 之前）用 "" 表示。"""
    secs = split_sections(body)
    bad: list[str] = []
    for u in verify.get("grounding", {}).get("ungrounded", []):
        frag = u["text"][:30]
        for h, txt in secs:
            if frag and frag in txt and h not in bad:
                bad.append(h)
    for name in required:
        if name not in body:
            bad.append(f"## {name}")
    return bad


def parse_patch(reply: str) -> dict[str, str]:
    """从模型回复中提取 {heading: section_markdown}。兼容 write_note JSON 或纯 Markdown。"""
    text = reply.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m and text.lstrip().startswith(("{", "```")):
        try:
            obj = json.loads(m.group(0))
            args = obj.get("args", obj)
            if isinstance(args.get("sections"), dict):
                return {k if k.startswith("##") else f"## {k}": v for k, v in args["sections"].items()}
            text = args.get("body_markdown", text)
        except Exception:  # noqa: BLE001
            pass
    text = re.sub(r"^```\w*\n|```\s*$", "", text.strip(), flags=re.M)
    return {h: txt for h, txt in split_sections(text) if h}


def _match_key(want: str, keys: list[str]) -> str | None:
    """标题模糊匹配：精确 > 前缀 > 包含；同分按文档顺序（keys 已按文档顺序排列）。"""
    if want in keys:
        return want
    for k in keys:
        if k.startswith(want) or want.startswith(k):
            return k
    for k in keys:
        if want in k or k in want:
            return k
    return None


def splice_sections(
    body: str,
    patch: dict[str, str],
    allowed_new: Iterable[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """把补丁按标题拼回正文。返回 (新正文, 已用补丁标题, 被丢弃的补丁标题)。

    标题模糊匹配（精确 > 前缀 > 包含）；匹配不上的补丁节直接丢弃并记入 dropped，
    不再追加到文末（避免补丁标题与正文标题措辞稍异时产生重复小节）。
    allowed_new 为空时不允许新增小节；传入标题列表时，只允许新增这些标题。
    """
    secs = split_sections(body)
    keys = [_key(h) if h else "" for h, _ in secs]
    by_key = {_key(h): h for h, _ in patch.items()}
    allowed = {_key(a) for a in allowed_new} if allowed_new is not None else set()
    used, dropped, out = [], [], []
    patched_idx: set[int] = set()
    for idx, (h, txt) in enumerate(secs):
        k = _key(h) if h else ""
        if h and k in by_key:
            ph = by_key[k]
            out.append((h, patch[ph].strip("\n")))
            used.append(ph)
            patched_idx.add(idx)
        else:
            out.append((h, txt))
    for ph, txt in patch.items():
        if ph in used:
            continue
        want = _key(ph)
        hit = _match_key(want, [k for k in keys if k])
        if hit is not None and keys.index(hit) not in patched_idx:
            idx = keys.index(hit)
            h, _ = out[idx]
            out[idx] = (h, txt.strip("\n"))
            used.append(ph)
            patched_idx.add(idx)
        elif want in allowed:
            out.append((ph, txt.strip("\n")))
            used.append(ph)
        else:
            dropped.append(ph)
    md = "\n".join((f"{h}\n{t}" if h else t).rstrip() + "\n" for h, t in out)
    return md.strip() + "\n", used, dropped
