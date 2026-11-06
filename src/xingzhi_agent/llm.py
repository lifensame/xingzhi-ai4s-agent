"""LLM 后端：offline 脚本化 / live OpenAI-compatible（书生）。"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import Settings


SYSTEM_PROMPT = """你是兴智杯 AI4S 科学智能体中的规划与写作模块。
你协助用户做「搜—读—算—写—辩—赛」闭环：文献检索、序列约束校验、实验笔记沉淀、证据辩证、假设锦标赛。
回答使用简体中文。不要声称代表任何官方竞赛组委会。
若需要调用工具，请只输出一个 JSON 对象（不要加解释文字），格式：
{"tool":"literature_search|sequence_check|design_experiment|write_note","args":{...}}
不要一次调用多个工具。写笔记时 args 需含 title 与 body_markdown。

工具说明：
- literature_search{query, top_k}：检索真实文献语料，返回带 id(Pxxx)/title/doi/url 的命中。
- sequence_check{sequence, molecule_type, assay_ph}：长度窗 + 理化指标（pI/GRAVY/净电荷/Cys/N-糖基化/CDR3）。
- design_experiment{sequence, molecule_type, assay_ph, target}：生成按判别价值排序的下一步实验方案。
- write_note{title, body_markdown}：写出 Markdown 实验笔记。

写作硬性要求（会被校验器打分，低于 90 分会被要求返工）：
1. 必含小节：意图、计划、证据、校验、风险，并写明下一步实验与 Go/No-Go；
2. 每条事实性主张（含数字、比较、因果、建议）后必须标注证据：
   文献用 [P001] 形式（只能用本轮检索命中的 ID），本轮工具计算结果用 [TOOL:sequence_check]
   或 [TOOL:design_experiment]；
3. 不得编造未命中的文献 ID，不得出现"官方指定/保证获奖/真实湿实验已测"等表述；
4. 笔记需注明为演示用途、与任何竞赛组委会无隶属关系。
"""


AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# S12：从用户目标文本中解析靶点（拉丁名）与候选序列
_TARGET_RES = [
    re.compile(r"围绕\s*([A-Za-z][A-Za-z0-9]{1,})\s*靶点"),
    re.compile(r"针对\s*([A-Za-z][A-Za-z0-9]{1,})\s*靶点?"),
    re.compile(r"([A-Z]{2,}[A-Z0-9]*)\s*靶点"),
]
_SEQ_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{40,}")


def _extract_sequence(goal: str) -> str | None:
    """从目标文本中找连续 ≥40 aa 的标准氨基酸串（去空白后匹配）。"""
    compact = re.sub(r"\s+", "", goal or "").upper()
    m = _SEQ_RE.search(compact)
    if m and 40 <= len(m.group()) <= 200:
        return m.group()
    return None


class OfflineLLM:
    """无 API 时的确定性脚本 LLM，保证 Demo 可复现。

    状态机：plan → literature_search → sequence_check → design_experiment → write_note
    笔记由真实工具结果拼装，每条事实性主张都带 [Pxxx] 引用，可通过 grounding 校验。
    S12：目标与序列可参数化（goal/sequence/target）；缺参数时默认 EGFR/VHH 演示目标，
    行为与原来完全一致。LIVE 模式（LiveLLM）按用户传入的 goal 规划，不走此脚本。
    """

    def __init__(self, goal: str | None = None, sequence: str | None = None,
                 target: str | None = None) -> None:
        self.turn = 0
        self.hits: list[dict[str, Any]] = []
        self.seq_data: dict[str, Any] = {}
        self.exp_data: dict[str, Any] = {}
        self.target = "EGFR"
        self.sequence = _demo_vhh_sequence()
        self.configure(goal, sequence, target)

    def configure(self, goal: str | None = None, sequence: str | None = None,
                  target: str | None = None) -> None:
        """从用户目标中解析靶点与候选序列（S12）。

        靶点：goal 里“围绕 X 靶点/针对 X”中的拉丁名；序列：goal 里连续 ≥40 aa 的
        标准氨基酸串。解析不到时保持当前值（默认 EGFR + 演示序列）。
        """
        if target:
            self.target = target.strip().upper()
        elif goal:
            for pat in _TARGET_RES:
                m = pat.search(goal)
                if m:
                    self.target = m.group(1).upper()
                    break
        if sequence:
            seq = re.sub(r"\s+", "", sequence).upper()
            if AA_RE.match(seq) and 40 <= len(seq) <= 200:
                self.sequence = seq
        elif goal:
            found = _extract_sequence(goal)
            if found:
                self.sequence = found

    # ---- 解析上一步工具结果 / parse the latest tool result ----
    def _absorb(self, last_user: str) -> None:
        try:
            blob = last_user.split("【工具结果】", 1)[1]
            payload = json.loads(blob[blob.index("{") : blob.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            return
        name, data = payload.get("tool"), payload.get("data") or {}
        if name == "literature_search":
            self.hits = data.get("hits", [])
        elif name == "sequence_check":
            self.seq_data = data
        elif name == "design_experiment":
            self.exp_data = data

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.turn += 1
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        if "【工具结果】" in last_user:
            self._absorb(last_user)
            if self.hits and not self.seq_data:
                return _tool_json("sequence_check", {
                    "sequence": self.sequence, "molecule_type": "VHH", "assay_ph": 6.0})
            if self.hits and self.seq_data and not self.exp_data:
                return _tool_json("design_experiment", {
                    "sequence": self.sequence, "molecule_type": "VHH",
                    "assay_ph": 6.0, "target": self.target})
            if self.hits and self.seq_data and self.exp_data:
                return _tool_json("write_note", {
                    "title": f"{self.target}_VHH_文献扫描_理化校验与下一步实验设计",
                    "body_markdown": compose_lab_note(self.hits, self.seq_data, self.exp_data, target=self.target),
                })

        if self.turn == 1:
            return (
                f"【意图理解】用户希望围绕 {self.target} 靶点的 VHH（纳米抗体）做文献扫描，"
                "核对长度与理化约束，并给出下一步实验与可复现笔记。\n"
                "【计划】1) 检索真实文献语料；2) 对候选序列做理化校验；"
                "3) 生成下一步实验设计；4) 汇总为带引用的 Markdown 实验笔记。"
            )

        if not self.hits:
            return _tool_json("literature_search", {
                "query": f"{self.target} VHH nanobody expression solubility pH binder design", "top_k": 5})
        return _tool_json("write_note", {
            "title": f"{self.target}_VHH_文献扫描_理化校验与下一步实验设计",
            "body_markdown": compose_lab_note(self.hits, self.seq_data, self.exp_data, target=self.target),
        })


def _tool_json(tool: str, args: dict[str, Any]) -> str:
    return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def _demo_vhh_sequence() -> str:
    # 119 aa 合法 VHH 占位序列（仅演示，非真实 binder）
    return (
        "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYY"
        "ADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYSYYFDYWGQGTLVTVSS"
    )


def _metric(seq_data: dict[str, Any], name: str, default: str = "—") -> Any:
    for m in seq_data.get("metrics", []):
        if m["name"] == name:
            return m["value"]
    return default


def compose_lab_note(
    hits: list[dict[str, Any]],
    seq_data: dict[str, Any],
    exp_data: dict[str, Any],
    target: str = "EGFR",
) -> str:
    """用真实工具结果拼装带引用的实验笔记 / assemble a fully cited lab note."""
    ph = seq_data.get("assay_ph", 6.0)
    # S1：兜底命中在证据表里显性标注（标题后加“默认推荐”），并在证据行下方加警告横幅
    fallback = any(h.get("fallback") for h in hits)
    ev_rows = "\n".join(
        f"| [{h['id']}] | {h['title'][:78]}{'（默认推荐）' if h.get('fallback') else ''} | {h.get('year','')} | "
        f"{('[' + (h.get('doi') or h.get('pmid') or 'link') + '](' + (h.get('url') or '') + ')') if h.get('url') else '—'} |"
        for h in hits
    ) or "| — | 无命中 | — | — |"

    claims = []
    for h in hits:
        kf = (h.get("key_findings") or [h.get("abstract", "")])[0]
        claims.append(f"- {kf[:200].strip()} [{h['id']}]")
    claim_block = "\n".join(claims) or "- 本轮无文献命中，不做任何事实性断言。"

    metric_rows = "\n".join(
        f"| {m['name']} | {m['value']}{(' ' + m['unit']) if m.get('unit') else ''} | "
        f"{m.get('threshold') or '—'} | {'✅' if m.get('ok') else ('❌' if m.get('ok') is False else '—')} | "
        f"{(m.get('basis') or '')[:46]} |"
        for m in seq_data.get("metrics", [])
    ) or "| — | — | — | — | — |"

    flags = seq_data.get("flags") or []
    flag_block = "\n".join(f"- ⚠ {f}" for f in flags) or "- 无额外风险提示。"

    exps = exp_data.get("ranked_experiments", [])
    exp_block = "\n\n".join(
        f"**优先级 {i}｜{e['experiment']}** [TOOL:design_experiment]\n\n"
        f"- 触发风险：{e['risk']} [TOOL:sequence_check]\n- 对照：{e['controls']}\n- 读出：{e['readout']} [TOOL:design_experiment]\n"
        f"- 判别价值：{e['discriminative_value']} [TOOL:design_experiment]\n- 成本：{e['cost']}"
        for i, e in enumerate(exps, 1)
    ) or "（本轮未生成实验设计）"

    ids = "、".join(f"[{h['id']}]" for h in hits) or "（无）"
    fallback_banner = (
        "\n> ⚠ 本轮无精确命中，以上文献为按年份默认推荐，不可作为检索证据引用；"
        "要点引用仅作格式演示，正式结论须以精确检索命中为准。"
        if fallback else ""
    )
    return f"""# 实验笔记：{target} VHH 文献扫描、理化校验与下一步实验设计

> 演示笔记 · 兴智杯 AI4S 科学智能体（搜—读—算—写—辩—赛）
> **声明**：本作品为参赛技术演示，与任何竞赛组委会或赛事主办方无隶属 / 赞助关系；示例序列为占位序列，不可用于真实湿实验。
> 本轮证据：{ids}（全部来自 Europe PMC 真实文献语料）{fallback_banner}

## 1. 意图

围绕 **{target}** 靶点的 **VHH** 候选，在设计前完成：真实文献扫描 → 理化约束校验 → 下一步实验设计 → 可复现笔记沉淀。

## 2. 计划（控制环）

意图 → 计划 → literature_search → sequence_check（理化指标）→ design_experiment → 校验器（含引用接地）→ write_note。

## 3. 证据表（本轮检索命中）

| 引用 | 标题 | 年份 | 来源 |
|------|------|------|------|
{ev_rows}

### 3.1 文献要点（每条主张均带引用）

{claim_block}

## 4. 理化校验（pH {ph} 测定条件）

| 指标 | 数值 | 阈值 | 判定 | 依据 |
|------|------|------|------|------|
{metric_rows}

序列指纹 `sha256:{seq_data.get('sequence_sha256', '—')}`，长度 {seq_data.get('length', '—')} aa，总体判定：{'通过' if seq_data.get('ok') else '未通过'} [TOOL:sequence_check]。

### 4.1 风险提示

{flag_block}

## 5. 下一步实验设计（按判别价值排序）

{exp_block}

**Go / No-Go** [TOOL:design_experiment]：{exp_data.get('go_no_go', '—')}

{exp_data.get('sample_size_note', '')} [TOOL:design_experiment]

## 6. 风险与自检

| 风险 | 自检手段 |
|------|----------|
| 长度或理化越界 | `sequence_check` 的阈值判定表 |
| 文献幻觉 | 仅引用本轮命中的真实文献 ID，校验器做引用-证据词面接地 |
| 过度承诺 | 校验器内置违禁表述黑名单（隶属声明 / 获奖承诺 / 湿实验结论） |
| 密钥泄漏 | 仓库仅含 `.env.example`；校验器扫描疑似密钥 |
| 不可复现 | 账本记录输入指纹、随机种子与产物哈希 |

## 7. 金句

**可表达、可复现、可追溯，高于单点高分。**
"""


class LLMError(RuntimeError):
    """LIVE 模型调用失败 / Live model call failed."""


class LiveLLM:
    """书生（InternLM）OpenAI 兼容接口客户端，带超时与指数退避重试。"""

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "缺少依赖 openai，请先 pip install -r requirements.txt，或改用 offline 模式。"
            ) from exc

        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,  # 自行控制重试节奏
        )
        self.model = settings.model
        self.seed = getattr(settings, "seed", 42)
        self.retries = max(0, settings.retries)
        self.backoff = 2.0
        self.last_usage: dict[str, int] = {}
        self.thinking = getattr(settings, "thinking", False)

    def chat(
        self,
        messages: list[dict[str, str]],
        timeout: float | None = None,
        max_tokens: int | None = None,
        retries: int | None = None,
    ) -> str:
        last_err: Exception | None = None
        n_retry = self.retries if retries is None else max(0, retries)
        for attempt in range(n_retry + 1):
            try:
                client = self.client.with_options(timeout=timeout) if timeout else self.client
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    seed=self.seed,
                    **({"max_tokens": int(max_tokens)} if max_tokens else {}),
                    # 关闭"深度思考"：实测同一请求 completion tokens 186→38、延迟 5.6s→1.6s
                    extra_body={"thinking_mode": bool(self.thinking)},
                )
                usage = getattr(resp, "usage", None)
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                last_err = RuntimeError("模型返回空内容")
            except Exception as exc:  # noqa: BLE001 - 统一收敛为 LLMError
                last_err = exc
            if attempt < n_retry:
                time.sleep(self.backoff * (attempt + 1))
        raise LLMError(f"{type(last_err).__name__}: {_redact(str(last_err))[:300]}")


def _redact(text: str) -> str:
    """脱敏：避免异常信息里带出 Key / Redact any key-like token。

    S17：除 sk-/eyJ/api_key 外，另覆盖 Bearer 明文、常见前缀族
    （ghp_/gho_/sk-proj-/xox*-）、URL 参数中的 token/secret、
    以及裸高熵 token（≥32 字符且同时含大小写与数字；纯小写 hex 摘要如
    note_sha256 不受影响，不会被误脱敏）。
    """
    text = re.sub(r"(sk-|eyJ)[A-Za-z0-9._\-]{8,}", "***REDACTED***", text)
    text = re.sub(r"(?i)(api[_-]?key\W{0,3})[A-Za-z0-9._\-]{8,}", r"\1***REDACTED***", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._\-]{16,}", "Bearer ***REDACTED***", text)
    text = re.sub(r"(ghp_|gho_|ghu_|ghs_|ghr_|sk-proj-|xox[baprs]-)[A-Za-z0-9]{8,}",
                  "***REDACTED***", text)
    text = re.sub(r"(?i)(token|secret|password)\s*=\s*[^&\s]{8,}", r"\1=***REDACTED***", text)

    def _bare(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if any(c.isupper() for c in tok) and any(c.islower() for c in tok) and any(c.isdigit() for c in tok):
            return "***REDACTED***"
        return tok

    return re.sub(r"[A-Za-z0-9_\-]{32,}", _bare, text)


def build_llm(settings: Settings) -> Any:
    if settings.is_live:
        return LiveLLM(settings)
    return OfflineLLM()


_TOOL_JSON_RE = re.compile(r"\{[\s\S]*\"tool\"[\s\S]*\}")

# 模型常在 Markdown 表格里把 | 写成 \|（如净电荷阈值 |q|），而 JSON 只认
# \" \\ \/ \b \f \n \r \t \uXXXX：\| 会让整段调用解析失败，主循环空转到
# max_steps（2026-11-05 两次 LIVE 实测因此失败）。严格解析失败时，去掉
# 非法转义的反斜杠再试一次（仅回退路径；合法 JSON 行为不变）。
_INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def _lenient_json_loads(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_INVALID_ESCAPE_RE.sub("", raw))
    except json.JSONDecodeError:
        return None


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """从模型输出中提取单个工具调用 JSON。"""
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    match = _TOOL_JSON_RE.search(text)
    if match:
        candidates.insert(0, match.group(0))

    for raw in candidates:
        obj = _lenient_json_loads(raw)
        if obj is None:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            args = obj.get("args") or obj.get("arguments") or obj.get("parameters")
            if not isinstance(args, dict):
                # 兼容扁平写法 {"tool":"x","query":...} / accept flat args
                args = {k: v for k, v in obj.items() if k not in ("tool", "args", "arguments", "parameters")}
            obj["args"] = args
            return obj
    return None
