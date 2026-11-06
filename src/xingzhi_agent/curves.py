"""实验曲线拟合 / 4-parameter logistic (4PL) dose–response fitting, stdlib only.

模型：y = bottom + (top - bottom) / (1 + (x / EC50) ** Hill)

拟合方法（确定性，无 scipy、无随机数）：
1. 粗网格：在 log10(EC50) × Hill 上穷举；对每一对 (EC50, Hill)，
   top/bottom 由线性最小二乘解析求解（模型在给定 EC50/Hill 时对
   bottom、span=top-bottom 是线性的），取 SSE 最小者；
2. 精修：围绕最优格点做 10 轮局部网格收缩（步长逐轮减半），
   EC50 相对精度约 0.02%。

适用与不适用（诚实标注）：
- 适用：单调的结合/抑制曲线（浓度递增、响应单调趋向平台），数据点 ≥ 5 且浓度全为正数。
- 不适用：双相曲线、拮抗剂多位点、时间序列、非单调机制数据——此时拟合仍会跑完
  但 R² 偏低并打 poor 标记，绝不静默给出"好"结果；EC50 落在数据浓度范围外时
  明确标记为外推，不可直接引用。
- 本模块不做权重拟合、不估参数置信区间；EC50 精度要求高的场景请用专业软件复核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["FitResult", "fit_4pl", "parse_curve_csv"]

_MIN_POINTS = 5
_HILL_RANGE = (0.3, 4.0)
_R2_GOOD = 0.95


@dataclass
class FitResult:
    """4PL 拟合结果（全部为浮点数/字符串，可直接 JSON 序列化）。"""

    ec50: float
    hill: float
    top: float
    bottom: float
    r2: float
    quality: str  # "good" | "poor"
    reasons: list[str] = field(default_factory=list)
    n_points: int = 0
    sse: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ec50": round(self.ec50, 4),
            "hill": round(self.hill, 3),
            "top": round(self.top, 4),
            "bottom": round(self.bottom, 4),
            "r2": round(self.r2, 4),
            "quality": self.quality,
            "reasons": list(self.reasons),
            "n_points": self.n_points,
            "sse": round(self.sse, 6),
        }


def _f_values(xs: list[float], ec50: float, hill: float) -> list[float]:
    """f = 1 / (1 + (x/EC50)^Hill)，对数域计算避免上溢。"""
    ln_ec50 = math.log(ec50)
    out = []
    for x in xs:
        z = hill * (math.log(x) - ln_ec50)
        if z > 500:
            out.append(0.0)
        elif z < -500:
            out.append(1.0)
        else:
            out.append(1.0 / (1.0 + math.exp(z)))
    return out


def _fit_top_bottom(xs: list[float], ys: list[float], ec50: float, hill: float) -> tuple[float, float, float]:
    """给定 (EC50, Hill)，线性最小二乘解 bottom 与 span=top-bottom，返回 (bottom, top, SSE)。"""
    fs = _f_values(xs, ec50, hill)
    n = len(xs)
    sf = sum(fs)
    sff = sum(f * f for f in fs)
    sy = sum(ys)
    sfy = sum(f * y for f, y in zip(fs, ys))
    det = n * sff - sf * sf
    if abs(det) < 1e-12:
        bottom = min(ys)
        top = max(ys)
    else:
        bottom = (sff * sy - sf * sfy) / det
        span = (n * sfy - sf * sy) / det
        top = bottom + span
    sse = sum((y - (bottom + (top - bottom) * f)) ** 2 for f, y in zip(fs, ys))
    return bottom, top, sse


def _validate(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    if len(points) < _MIN_POINTS:
        raise ValueError(
            f"数据点太少：4 参数模型至少需要 {_MIN_POINTS} 个点，当前 {len(points)} 个"
        )
    xs, ys = [], []
    for i, (x, y) in enumerate(points):
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            raise ValueError(f"第 {i + 1} 个点不是数值：({x!r}, {y!r})") from None
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise ValueError(f"第 {i + 1} 个点含非有限数值：({x!r}, {y!r})")
        if xf <= 0:
            raise ValueError(f"浓度必须为正数：第 {i + 1} 个点浓度为 {xf}（4PL 在对数浓度域拟合）")
        xs.append(xf)
        ys.append(yf)
    return xs, ys


def fit_4pl(points: list[tuple[float, float]]) -> FitResult:
    """拟合 4PL 曲线。

    入参：[(浓度, 响应), ...]，浓度单位由调用方约定（如 nM），EC50 返回同单位。
    返回 FitResult；数据有问题时抛 ValueError（不静默返回假结果）。
    非单调/噪声大的数据不会崩溃，只会得到低 R² 与 poor 标记。
    """
    xs, ys = _validate(points)
    n = len(xs)
    lo = math.log10(min(xs)) - 0.5
    hi = math.log10(max(xs)) + 0.5

    best: tuple[float, float, float, float, float] | None = None  # (sse, log_ec50, hill, bottom, top)

    def _scan(log_ec50: float, hill: float) -> None:
        nonlocal best
        ec50 = 10 ** log_ec50
        bottom, top, sse = _fit_top_bottom(xs, ys, ec50, hill)
        if best is None or sse < best[0]:
            best = (sse, log_ec50, hill, bottom, top)

    # 1) 粗网格：logEC50 步长 0.1，Hill 0.2–4.0 步长 0.1
    steps = int(round((hi - lo) / 0.1)) + 1
    for i in range(steps):
        log_ec50 = lo + i * 0.1
        h = 0.2
        while h <= 4.0 + 1e-9:
            _scan(log_ec50, round(h, 3))
            h += 0.1

    # 2) 精修：10 轮局部 3×3 网格，步长逐轮减半
    assert best is not None
    step_log, step_hill = 0.05, 0.05
    for _ in range(10):
        _, bl, bh, _, _ = best
        for dlog in (-step_log, 0.0, step_log):
            for dh in (-step_hill, 0.0, step_hill):
                log_ec50 = bl + dlog
                hill = min(6.0, max(0.1, bh + dh))
                _scan(log_ec50, hill)
        step_log /= 2
        step_hill /= 2

    sse, log_ec50, hill, bottom, top = best
    ec50 = 10 ** log_ec50
    mean_y = sum(ys) / n
    sst = sum((y - mean_y) ** 2 for y in ys)
    if sst <= 0:
        r2 = 1.0 if sse <= 1e-12 else 0.0
    else:
        r2 = max(0.0, 1.0 - sse / sst)

    reasons: list[str] = []
    if r2 < _R2_GOOD:
        reasons.append(f"R²={r2:.3f} 低于 {_R2_GOOD}，拟合不可靠")
    if ec50 < min(xs) or ec50 > max(xs):
        reasons.append(
            f"EC50={ec50:.4g} 超出数据浓度范围 [{min(xs):.4g}, {max(xs):.4g}]，属于外推"
        )
    if hill < _HILL_RANGE[0] or hill > _HILL_RANGE[1]:
        reasons.append(f"Hill={hill:.2f} 超出常规范围 [{_HILL_RANGE[0]}, {_HILL_RANGE[1]}]")
    if top <= bottom:
        reasons.append("拟合为递减曲线（top ≤ bottom），请确认是抑制型数据")
    return FitResult(
        ec50=ec50, hill=hill, top=top, bottom=bottom, r2=r2,
        quality="good" if not reasons else "poor",
        reasons=reasons, n_points=n, sse=sse,
    )


def parse_curve_csv(text: str) -> list[tuple[float, float]]:
    """解析粘贴的浓度–响应文本。

    接受逗号 / 制表符 / 分号 / 空白分隔；首行若为表头
    （concentration, response / 浓度, 响应 等非数值行）自动跳过；
    其余每行须恰好 2 列数值。返回按浓度升序的 [(浓度, 响应), ...]。
    """
    if not text or not text.strip():
        raise ValueError("曲线数据为空：请粘贴两列数据（浓度, 响应）")
    points: list[tuple[float, float]] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    for idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        elif "\t" in raw:
            parts = [p.strip() for p in raw.split("\t")]
        elif ";" in raw:
            parts = [p.strip() for p in raw.split(";")]
        else:
            parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"第 {idx} 行应为 2 列，实际 {len(parts)} 列：{raw[:60]!r}")
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            if idx == 1 and not points:
                # 首行非数值 → 当作表头跳过（concentration,response / 浓度,响应等）
                continue
            raise ValueError(f"第 {idx} 行不是数值：{raw[:60]!r}") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"第 {idx} 行含非有限数值：{raw[:60]!r}")
        if x <= 0:
            raise ValueError(f"第 {idx} 行浓度必须为正数：{x}")
        points.append((x, y))
    if len(points) < _MIN_POINTS:
        raise ValueError(
            f"数据点太少：4 参数模型至少需要 {_MIN_POINTS} 个点，当前 {len(points)} 个"
        )
    return sorted(points)
