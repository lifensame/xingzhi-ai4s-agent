"""成本 / 可复现账本 / Cost & reproducibility ledger.

创新点 ⑤：记录每次 LLM 调用的延迟与 token 用量，估算成本；并生成
"输入指纹 + 产物哈希"的可复现清单，评委可用同一输入验证产出一致性。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 书生开放平台当前对个人额度免费；单价可按需覆盖（元 / 千 token）
PRICE_PER_1K_IN = 0.0
PRICE_PER_1K_OUT = 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Call:
    step: int
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    ok: bool = True
    prompt_sha256: str = ""
    phase: str = "agent"


@dataclass
class Ledger:
    mode: str
    model: str
    calls: list[Call] = field(default_factory=list)
    tool_calls: int = 0
    tool_retries: int = 0
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def add_call(self, **kw: Any) -> None:
        self.calls.append(Call(**kw))

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def cost_cny(self) -> float:
        cin = sum(c.prompt_tokens for c in self.calls) / 1000 * PRICE_PER_1K_IN
        cout = sum(c.completion_tokens for c in self.calls) / 1000 * PRICE_PER_1K_OUT
        return round(cin + cout, 6)

    def manifest(self) -> dict[str, Any]:
        lat = [c.latency_s for c in self.calls]
        return {
            "mode": self.mode,
            "model": self.model,
            "llm_calls": len(self.calls),
            "tool_calls": self.tool_calls,
            "tool_retries": self.tool_retries,
            "retry_events": self.retry_events,
            "total_tokens": self.total_tokens,
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cost_cny_estimate": self.cost_cny,
            "latency_s": {
                "total_wall": round(time.time() - self.started, 2),
                "llm_sum": round(sum(lat), 2),
                "llm_max": round(max(lat), 2) if lat else 0.0,
                "llm_avg": round(sum(lat) / len(lat), 2) if lat else 0.0,
            },
            "steps": [c.__dict__ for c in self.calls],
            "inputs": self.inputs,
            "outputs": self.outputs,
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            # O8：确定性产物与本次运行计时拆节——"reproducible" 不再是 mode=="offline"
            # 的裸静态断言；determinism 节只含可比对的哈希（两次离线运行应完全相同），
            # timing 节的墙钟每次都变，不与“可复现”混在同一叙事里。
            "determinism": {
                "note_sha256": self.outputs.get("note_sha256"),
                "claim": ("offline deterministic script: same inputs -> same note_sha256"
                          if self.mode == "offline" else
                          "live: inputs reproducible, outputs sampled"),
                "verified_by": "compare determinism.note_sha256 across two runs (see scripts/verify_all.sh)",
            },
            "timing": {
                "total_wall_s": round(time.time() - self.started, 2),
                "llm_sum_s": round(sum(lat), 2),
            },
            "reproducible": {
                "deterministic_script": self.mode == "offline",
                "basis": "mode == offline (static claim); verify by comparing determinism.note_sha256 across runs",
                "note_sha256": self.outputs.get("note_sha256"),
            },
            "note": "offline 模式为确定性脚本 LLM：相同输入 → 相同 note_sha256；"
                    "live 模式受模型采样影响，仅输入指纹可复现。",
        }

    def dump(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
