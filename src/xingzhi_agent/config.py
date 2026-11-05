"""环境配置：默认离线；有 Key + Base URL 时切 LIVE。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    # 离线 Demo 允许无第三方依赖；手动解析简单 KEY=VALUE
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in __import__("os").environ:
                __import__("os").environ[k] = v


@dataclass(frozen=True)
class Settings:
    demo_mode: str  # offline | live
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout: float
    retries: int
    corpus_path: Path
    outputs_dir: Path
    seed: int = 42
    repair_timeout: float = 90.0
    repair_budget: float = 180.0      # 整个修复阶段的硬墙钟预算（秒）
    repair_max_tokens: int = 1200     # 单次修复回复上限
    agent_max_tokens: int = 3000      # 主循环单步回复上限
    thinking: bool = False            # 书生 thinking_mode（默认关，显著降延迟）

    @property
    def is_live(self) -> bool:
        return (
            self.demo_mode == "live"
            and bool(self.api_key.strip())
            and bool(self.base_url.strip())
        )


def _env_float(name: str, default: float) -> float:
    """环境变量解析容错：非法值打印告警并回落默认值，不抛原生 ValueError。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        print(f"⚠ 环境变量 {name}={raw!r} 不是合法数字，已回落默认值 {default}")
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def load_settings() -> Settings:
    mode = os.getenv("DEMO_MODE", "offline").strip().lower()
    api_key = os.getenv("INTERNLM_API_KEY", "").strip()
    if not api_key:
        # Streamlit Cloud 部署：密钥填在应用 Settings → Secrets
        #（.streamlit/secrets.toml），不经过 .env；本地/CLI 无 secrets
        # 时静默跳过，仍走离线。环境变量优先，Secrets 只作回退。
        try:
            import streamlit as st

            api_key = str(st.secrets.get("INTERNLM_API_KEY", "") or "").strip()
        except Exception:
            pass
    base_url = (os.getenv("INTERNLM_BASE_URL", "").strip() or "https://chat.intern-ai.org.cn/api/v1")
    model = (os.getenv("INTERNLM_MODEL", "").strip() or "intern-s2")
    # 默认 120s：实测 LIVE 最慢一步（写整篇笔记）32.5s，120s 是约 3.7 倍余量；
    # 60s 太紧（推理模型写长笔记可能超 60s），300s 则让挂住的接口在 UI 上
    # 静默卡住过久。网络较慢时可用 INTERNLM_TIMEOUT 显式调大。
    timeout = _env_float("INTERNLM_TIMEOUT", 120)
    retries = _env_int("INTERNLM_RETRIES", 2)

    # S17：LIVE 必须显式 DEMO_MODE=live；仅有 Key 不再自动升级，
    # 避免残留 .env 让“离线复现”悄悄变成联网调用。
    if mode == "live" and (not api_key or not base_url):
        print("⚠ DEMO_MODE=live 但缺少 INTERNLM_API_KEY 或 INTERNLM_BASE_URL，已回落 offline")
        mode = "offline"

    return Settings(
        demo_mode=mode,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        corpus_path=ROOT / "data" / "corpus.json",
        outputs_dir=ROOT / "outputs",
        seed=_env_int("XZ_SEED", 42),
        repair_timeout=_env_float("XZ_REPAIR_TIMEOUT", 90),
        repair_budget=_env_float("XZ_REPAIR_BUDGET", 180),
        repair_max_tokens=_env_int("XZ_REPAIR_MAX_TOKENS", 1200),
        agent_max_tokens=_env_int("XZ_AGENT_MAX_TOKENS", 3000),
        thinking=os.getenv("XZ_THINKING", "0").strip().lower() in ("1", "true", "yes", "on"),
    )
