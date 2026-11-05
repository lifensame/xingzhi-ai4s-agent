"""MCP server（stdio）：把 tools.py 的三个科学计算工具原样包成 MCP tools（P2-2）。

薄封装，不重写工具逻辑，不改现有调用路径，离线 demo 不依赖本模块。
基于官方 MCP Python SDK v2（mcp==2.3.0）：v2 把 FastMCP 更名为 MCPServer
（`from mcp.server.mcpserver import MCPServer`），用法与 v1 的 FastMCP 相同：
`@mcp.tool()` 注册工具，`mcp.run(transport="stdio")` 启动。

启动：
    python scripts/run_mcp_server.py [--corpus data/corpus.json]
客户端配置示例（Claude Desktop / 任意 MCP 客户端）：
    {"mcpServers": {"xingzhi-ai4s": {"command": "python",
     "args": ["scripts/run_mcp_server.py"], "cwd": "<包根>"}}}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from xingzhi_agent import tools as _tools  # noqa: E402

mcp = MCPServer(
    name="xingzhi-ai4s-agent",
    description="兴智杯 AI4S 科学智能体：文献检索、序列理化校验、实验设计（薄封装，不改工具逻辑）",
)

_CORPUS: list[dict[str, Any]] = []


def load_corpus_for_server(corpus_path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载语料（server 启动时调一次；失败时 literature_search 返回明确错误而非崩溃）。"""
    global _CORPUS
    path = Path(corpus_path or os.getenv("XZ_CORPUS_PATH", "") or (ROOT / "data" / "corpus.json"))
    try:
        _CORPUS = _tools.load_corpus(path)
    except Exception as exc:  # noqa: BLE001 - 语料缺失时 server 仍可启动，检索工具会报错说明
        _CORPUS = []
        print(f"⚠ MCP server 语料加载失败（{path}）：{exc}，literature_search 将返回错误", file=sys.stderr)
    return _CORPUS


def _dump(result: _tools.ToolResult) -> dict[str, Any]:
    """工具返回值与现有 ToolResult.data 结构一致（ok / summary / data）。"""
    return {"ok": result.ok, "name": result.name, "summary": result.summary, "data": result.data}


@mcp.tool(
    description="在本地纳米抗体/VHH 文献语料（52 篇，Europe PMC 真实摘要）中检索；"
                "无精确命中时返回按年份默认推荐并打 fallback 标记，不得作为检索证据引用。"
)
def literature_search(query: str, top_k: int = 4) -> dict[str, Any]:
    if not _CORPUS:
        return {"ok": False, "name": "literature_search",
                "summary": "文献检索失败：语料未加载",
                "data": {"hits": [], "query": query, "fallback": False}}
    return _dump(_tools.literature_search(_CORPUS, query, top_k=top_k))


@mcp.tool(
    description="VHH/minibinder 序列理化校验：长度窗、等电点 pI、GRAVY、净电荷、CDR3、"
                "ε280、N-糖基化等指标。assay_ph 必须在 [0, 14] 内；未知 molecule_type 明确报错。"
)
def sequence_check(sequence: str, molecule_type: str = "VHH", assay_ph: float = 6.0) -> dict[str, Any]:
    return _dump(_tools.sequence_check(sequence, molecule_type, assay_ph))


@mcp.tool(
    description="基于序列理化结果设计下一步判别性实验（SEC-HPLC/PSR/BLI 等），"
                "返回排序后的实验方案与 go/no-go 判据。assay_ph 必须在 [0, 14] 内。"
)
def design_experiment(
    sequence: str, molecule_type: str = "VHH", assay_ph: float = 6.0, target: str = "EGFR",
) -> dict[str, Any]:
    return _dump(_tools.design_experiment(sequence, molecule_type, assay_ph, target))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="兴智杯 AI4S 科学智能体 MCP server（stdio）")
    ap.add_argument("--corpus", default=None, help="语料路径（默认 data/corpus.json，可用 XZ_CORPUS_PATH 覆盖）")
    args = ap.parse_args()
    load_corpus_for_server(args.corpus)
    print(f"MCP server 已启动（stdio）· 语料 {len(_CORPUS)} 篇 · 工具 3 个", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
