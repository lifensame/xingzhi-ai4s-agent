"""分子结构解析（SMILES 限定子集）/ SMILES parser + descriptors, stdlib only.

支持的 SMILES 子集（超出即明确报错，绝不静默按别的意思解析）：
- 元素：C、N、O、S、P、F、Cl、Br，及芳香小写 c、n、o、s、p；括号原子如 [O-]、[NH4+]、[nH]（元素 + H 数 + 电荷）。
- 键：单键（隐式或 -）、双键 =、三键 #、芳香键 : 或 ~。
- 分支 ( )、环闭合数字 1–9（同一数字出现两次即成环）。
- 电荷：括号原子上的 + / -（如 [O-]、[N+]）。

明确不支持（遇见即 ValueError）：
- 立体化学：/、\\、@（手性/顺反异构不解析）；
- %nn 双位数环闭合（只支持 1–9）；
- 多组分 .（盐/混合物不合并计算）；
- 上述元素之外的元素（如 B、Si、I、金属、[Na+] 等）与同位素（如 [13C]）；
- 芳香硒等罕见芳香原子。

描述符与精度说明：
- 分子式（Hill 体系）与分子量：精确计数，咖啡因/阿司匹林与公布值一致（见单测）。
- HBD：带 H 的 O/N 数；HBA：全部 O + 电荷 ≤ 0 的 N（与 RDKit 的 Lipinski 计数口径一致）。
- 可旋转键：非环单键、两端均非末端原子，排除酰胺 C–N（部分双键性）。
- TPSA：Ertl 片段贡献（OH 20.23 / =O 17.07 / 醚氧 9.23 / NH2 26.02 / NH 12.03 /
  叔胺 N 3.24 / 芳香 nH 15.79 / 芳香 n 12.89 / 硝基 N 11.68 / 硫醇 S 25.30）。
  咖啡因估算 66.4 Å²，PubChem 公布 61.82 Å²，偏差来自芳香 N-甲基化的归类差异；
  阿司匹林 63.60、乙醇 20.23 与公布值一致。TPSA 是估算值，不用于合规判定。
- logP：简化原子贡献估算（类 Crippen，见 _LOGP），近似值，误差约 ±0.5，
  仅供参考；Lipinski 判定会注明 logP 为估算值。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = ["Molecule", "parse_smiles", "describe"]

ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904,
}

# 中性非芳香原子的价态（键级和上限）
_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1}
# 芳香原子按"连接数"计隐式 H：芳香 C 最多 3 个连接，芳香 n/o/s/p 无隐式 H（[nH] 除外，括号显式给 H）
_AROMATIC_ELEMENTS = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P"}
_ORGANIC_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br"}
_BOND_ORDER = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "~": 1.5}

# Ertl TPSA 片段贡献（Å²）
_TPSA_OH = 20.23
_TPSA_CARBONYL_O = 17.07
_TPSA_ETHER_O = 9.23
_TPSA_NH2 = 26.02
_TPSA_NH = 12.03
_TPSA_N_TERTIARY = 3.24
_TPSA_AROMATIC_NH = 15.79
_TPSA_AROMATIC_N = 12.89
_TPSA_NITRO_N = 11.68
_TPSA_THIOL_S = 25.30

# 简化 logP 原子贡献（类 Crippen，近似值；三分子校准：咖啡因 -0.10/-0.07、
# 阿司匹林 0.90/1.19、乙醇 0.05/-0.31，误差 < 0.5）
_LOGP_AROMATIC_C = 0.30
_LOGP_AROMATIC_C_HETERO = 0.15
_LOGP_SP3_C = 0.40
_LOGP_SP3_C_HETERO = 0.10


@dataclass
class _Atom:
    element: str  # 大写元素符号
    aromatic: bool = False
    charge: int = 0
    h_explicit: int | None = None  # 括号原子显式给的 H 数；None 表示按规则算隐式 H
    neighbors: list[tuple[int, float]] = field(default_factory=list)  # (原子下标, 键级)

    @property
    def h_count(self) -> int:
        if self.h_explicit is not None:
            return self.h_explicit
        n_conn = len(self.neighbors)
        if self.aromatic:
            # 芳香 C：3 个连接位，缺的补 H；芳香 n/o/s/p：无隐式 H（吡咯型 N 须写 [nH]）
            if self.element == "C":
                return max(0, 3 - n_conn)
            return 0
        bond_sum = sum(b for _, b in self.neighbors)
        if self.element == "C":
            return max(0, 4 - int(round(bond_sum)))
        if self.element == "N":
            valence = 4 if self.charge > 0 else (2 if self.charge < 0 else 3)
            return max(0, valence - int(round(bond_sum)))
        if self.element == "O":
            valence = 1 if self.charge < 0 else 2
            return max(0, valence - int(round(bond_sum)))
        if self.element == "S":
            return max(0, 2 - int(round(bond_sum)))
        if self.element == "P":
            return max(0, 3 - int(round(bond_sum)))
        return 0  # 卤素无隐式 H


@dataclass
class Molecule:
    """解析后的分子：原子表 + 键表（下标引用）。"""

    atoms: list[_Atom]
    smiles: str = ""

    # -- 基础量 --
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for a in self.atoms:
            counts[a.element] = counts.get(a.element, 0) + 1
            if a.h_count:
                counts["H"] = counts.get("H", 0) + a.h_count
        if not counts:
            return ""
        parts: list[str] = []
        for el in ("C", "H"):
            if el in counts:
                parts.append(el + (str(counts[el]) if counts[el] != 1 else ""))
        for el in sorted(k for k in counts if k not in ("C", "H")):
            parts.append(el + (str(counts[el]) if counts[el] != 1 else ""))
        return "".join(parts)

    def molecular_weight(self) -> float:
        total = 0.0
        for a in self.atoms:
            total += ATOMIC_WEIGHTS[a.element] + a.h_count * ATOMIC_WEIGHTS["H"]
        return round(total, 2)

    def hbd(self) -> int:
        """H 键供体：带 H 的 O/N 原子数。"""
        return sum(1 for a in self.atoms if a.element in ("O", "N") and a.h_count > 0)

    def hba(self) -> int:
        """H 键受体：全部 O + 电荷 ≤ 0 的 N（季铵 N+ 不算）。"""
        return sum(
            1 for a in self.atoms
            if a.element == "O" or (a.element == "N" and a.charge <= 0)
        )

    def _in_ring(self, i: int, j: int) -> bool:
        """去掉键 (i, j) 后两端是否仍连通 → 该键是否在环上（BFS）。"""
        seen = {i}
        stack = [i]
        while stack:
            cur = stack.pop()
            for nb, _ in self.atoms[cur].neighbors:
                if (cur == i and nb == j) or (cur == j and nb == i):
                    continue
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return j in seen

    def rotatable_bonds(self) -> int:
        """可旋转键：非环单键，两端均非末端原子，排除酰胺 C–N。"""
        count = 0
        seen: set[tuple[int, int]] = set()
        for i, a in enumerate(self.atoms):
            for j, order in a.neighbors:
                if (j, i) in seen or order != 1.0:
                    continue
                seen.add((i, j))
                if len(self.atoms[i].neighbors) < 2 or len(self.atoms[j].neighbors) < 2:
                    continue  # 末端原子相连的键不旋转
                if self._in_ring(i, j):
                    continue
                # 酰胺 C–N：C 连着一个 =O 且另一端是 N，视为部分双键，不计
                if self._is_amide_bond(i, j):
                    continue
                count += 1
        return count

    def _is_amide_bond(self, i: int, j: int) -> bool:
        for c_idx, n_idx in ((i, j), (j, i)):
            c, n = self.atoms[c_idx], self.atoms[n_idx]
            if c.element == "C" and not c.aromatic and n.element == "N":
                if any(nb != n_idx and self.atoms[nb].element == "O"
                       and order == 2.0 for nb, order in c.neighbors):
                    return True
        return False

    def _is_amide_like_n(self, a: _Atom) -> bool:
        """酰胺型 N：连着一个带 =O 的碳（如咖啡因的两个尿素型 N、酰胺 N）。

        咖啡因 SMILES 里这两个 N 写成芳香小写 n，但化学上是酰胺型：
        TPSA 按叔胺 3.24 计（而非芳香 n 的 12.89），否则咖啡因 TPSA 会算到 85.7，
        与 PubChem 公布 61.82 偏差过大；修正后为 66.4。
        """
        for nb, _ in a.neighbors:
            c = self.atoms[nb]
            if c.element != "C":
                continue
            if any(self.atoms[nn].element == "O" and o == 2.0
                   for nn, o in c.neighbors):
                return True
        return False

    def tpsa(self) -> float:
        """Ertl 片段贡献求和（估算值，见模块 docstring 的精度说明）。"""
        total = 0.0
        for a in self.atoms:
            if a.element == "O":
                if a.charge < 0:
                    total += _TPSA_CARBONYL_O
                elif a.h_count > 0:
                    total += _TPSA_OH
                elif any(order == 2.0 for _, order in a.neighbors):
                    total += _TPSA_CARBONYL_O
                else:
                    total += _TPSA_ETHER_O
            elif a.element == "N":
                if a.charge > 0:
                    continue  # 季铵 N 无孤对，不贡献
                if a.aromatic:
                    if self._is_amide_like_n(a):
                        total += _TPSA_N_TERTIARY
                    else:
                        total += _TPSA_AROMATIC_NH if a.h_count > 0 else _TPSA_AROMATIC_N
                elif self._is_nitro_n(a):
                    total += _TPSA_NITRO_N
                elif a.h_count == 2:
                    total += _TPSA_NH2
                elif a.h_count == 1:
                    total += _TPSA_NH
                else:
                    total += _TPSA_N_TERTIARY
            elif a.element == "S":
                if a.h_count > 0:
                    total += _TPSA_THIOL_S
        return round(total, 2)

    def _is_nitro_n(self, a: _Atom) -> bool:
        o_double = sum(1 for nb, order in a.neighbors
                       if self.atoms[nb].element == "O" and order == 2.0)
        return o_double >= 1 and not a.aromatic

    def logp_estimate(self) -> float:
        """简化原子贡献 logP 估算（近似值，误差约 ±0.5，仅供参考）。"""
        total = 0.0
        for a in self.atoms:
            el = a.element
            if el == "C":
                if a.aromatic:
                    has_hetero = any(self.atoms[nb].element != "C" for nb, _ in a.neighbors)
                    total += _LOGP_AROMATIC_C_HETERO if has_hetero else _LOGP_AROMATIC_C
                elif any(order == 2.0 for _, order in a.neighbors):
                    total += 0.0  # 羰基/烯碳
                else:
                    has_hetero = any(self.atoms[nb].element != "C" for nb, _ in a.neighbors)
                    total += _LOGP_SP3_C_HETERO if has_hetero else _LOGP_SP3_C
            elif el == "N":
                if a.charge > 0:
                    total -= 2.00
                elif a.aromatic:
                    # 酰胺型芳香 N（如咖啡因尿素 N）与三连接芳香 N（如 N-甲基咪唑 N）：
                    # 孤对离域，贡献小于吡啶型 N
                    if self._is_amide_like_n(a) or len(a.neighbors) >= 3:
                        total -= 0.15
                    else:
                        total += -0.50 if a.h_count > 0 else -0.25
                elif a.h_count == 2:
                    total -= 1.00
                elif a.h_count == 1:
                    total -= 0.75
                else:
                    total -= 0.15
            elif el == "O":
                if a.charge < 0:
                    total -= 1.50
                elif a.h_count > 0:
                    total -= 0.45
                elif any(order == 2.0 for _, order in a.neighbors):
                    total -= 0.25
                else:
                    total -= 0.20
            elif el == "S":
                total += 0.20
            elif el == "F":
                total += 0.30
            elif el == "Cl":
                total += 0.50
            elif el == "Br":
                total += 0.70
            elif el == "P":
                total -= 0.50
        return round(total, 2)

    def lipinski(self) -> dict:
        mw = self.molecular_weight()
        logp = self.logp_estimate()
        hbd = self.hbd()
        hba = self.hba()
        criteria = [
            {"rule": "分子量 ≤ 500 Da", "value": mw, "pass": mw <= 500},
            {"rule": "logP ≤ 5（简化估算值）", "value": logp, "pass": logp <= 5},
            {"rule": "H 键供体 ≤ 5", "value": hbd, "pass": hbd <= 5},
            {"rule": "H 键受体 ≤ 10", "value": hba, "pass": hba <= 10},
        ]
        violations = [c["rule"] for c in criteria if not c["pass"]]
        return {
            "verdict": "通过" if not violations else f"违反 {len(violations)} 项",
            "violations": violations,
            "criteria": criteria,
            "note": "logP 为简化原子贡献估算值（误差约 ±0.5），Lipinski 判定仅供参考，不作合规结论",
        }

    def as_dict(self) -> dict:
        lip = self.lipinski()
        return {
            "smiles": self.smiles,
            "formula": self.formula(),
            "molecular_weight": self.molecular_weight(),
            "hbd": self.hbd(),
            "hba": self.hba(),
            "rotatable_bonds": self.rotatable_bonds(),
            "tpsa": self.tpsa(),
            "logp_estimate": self.logp_estimate(),
            "lipinski": lip,
            "n_heavy_atoms": len(self.atoms),
        }


_BRACKET_RE = re.compile(
    r"^(?:(\d+))?"          # 同位素数字（不支持，遇见即报错）
    r"([A-Z][a-z]?|[a-z])"  # 元素
    r"(?:H(\d*))?"          # 显式 H 及个数
    r"([+-](?:\d+)?|[+-]{2})?$"  # 电荷
)


def _parse_bracket(content: str, pos: int) -> _Atom:
    if "@" in content:
        raise ValueError(f"不支持立体化学 @（位置 {pos}）：手性标记不在支持子集内")
    m = _BRACKET_RE.match(content)
    if not m:
        raise ValueError(f"无法解析括号原子 [{content}]（位置 {pos}）")
    isotope, element, h_digits, charge_s = m.groups()
    if isotope:
        raise ValueError(f"不支持同位素 [{content}]（位置 {pos}）")
    aromatic = element.islower()
    el = _AROMATIC_ELEMENTS.get(element, element)
    if el not in _ORGANIC_ELEMENTS:
        raise ValueError(
            f"不支持的元素 {element!r}（位置 {pos}）：本模块仅支持 C、N、O、S、P、F、Cl、Br"
        )
    h_explicit = None
    if h_digits is not None:
        h_explicit = int(h_digits) if h_digits else 1
    charge = 0
    if charge_s:
        if charge_s in ("+", "++"):
            charge = 1 if charge_s == "+" else 2
        elif charge_s in ("-", "--"):
            charge = -1 if charge_s == "-" else -2
        else:
            charge = int(charge_s)
    return _Atom(element=el, aromatic=aromatic, charge=charge, h_explicit=h_explicit)


def parse_smiles(smiles: str) -> Molecule:
    """解析 SMILES（限定子集，见模块 docstring），返回 Molecule。

    非法输入抛 ValueError，信息含位置与原因；绝不静默按别的意思解析。
    """
    if not smiles or not str(smiles).strip():
        raise ValueError("SMILES 为空")
    s = str(smiles).strip()
    atoms: list[_Atom] = []
    current: int | None = None
    stack: list[int | None] = []
    ring_open: dict[str, tuple[int, float]] = {}
    pending_bond: float | None = None
    i = 0
    n = len(s)

    def _default_bond(a: _Atom, b: _Atom) -> float:
        return 1.5 if (a.aromatic and b.aromatic) else 1.0

    def _add_bond(i1: int, i2: int, order: float) -> None:
        atoms[i1].neighbors.append((i2, order))
        atoms[i2].neighbors.append((i1, order))

    while i < n:
        ch = s[i]
        if ch in "-=#:~":
            pending_bond = _BOND_ORDER[ch]
            i += 1
            continue
        if ch in "/\\":
            raise ValueError(f"不支持立体化学 {ch!r}（位置 {i}）：顺反/手性标记不在支持子集内")
        if ch == "@":
            raise ValueError(f"不支持立体化学 @（位置 {i}）：手性标记不在支持子集内")
        if ch == "%":
            raise ValueError(f"不支持 %nn 双位数环闭合（位置 {i}）：环编号仅支持 1–9")
        if ch == ".":
            raise ValueError(f"不支持多组分 .（位置 {i}）：盐/混合物不合并计算")
        if ch == "(":
            stack.append(current)
            i += 1
            continue
        if ch == ")":
            if not stack:
                raise ValueError(f"括号不匹配：位置 {i} 的 ) 没有对应的 (")
            current = stack.pop()
            i += 1
            continue
        if ch.isdigit():
            if current is None:
                raise ValueError(f"环闭合数字 {ch!r}（位置 {i}）前没有原子")
            if ch in ring_open:
                other, stored_bond = ring_open.pop(ch)
                order = pending_bond if pending_bond is not None else stored_bond
                _add_bond(current, other, order)
            else:
                # 未指定键级时：芳香原子默认芳香键 1.5，否则单键
                bond = pending_bond if pending_bond is not None else (
                    1.5 if atoms[current].aromatic else 1.0
                )
                ring_open[ch] = (current, bond)
            pending_bond = None
            i += 1
            continue
        # 原子
        if ch == "[":
            j = s.find("]", i)
            if j < 0:
                raise ValueError(f"括号原子未闭合（位置 {i}）：缺少 ]")
            atom = _parse_bracket(s[i + 1:j], i)
            i = j + 1
        elif ch in _AROMATIC_ELEMENTS:
            atom = _Atom(element=_AROMATIC_ELEMENTS[ch], aromatic=True)
            i += 1
        elif s[i:i + 2] == "Cl":
            atom = _Atom(element="Cl")
            i += 2
        elif s[i:i + 2] == "Br":
            atom = _Atom(element="Br")
            i += 2
        elif ch in _ORGANIC_ELEMENTS:
            atom = _Atom(element=ch)
            i += 1
        else:
            raise ValueError(
                f"不支持的字符 {ch!r}（位置 {i}）：本模块仅支持 C、N、O、S、P、F、Cl、Br"
                "及芳香小写 c、n、o、s、p"
            )
        idx = len(atoms)
        atoms.append(atom)
        if current is not None:
            order = pending_bond if pending_bond is not None else _default_bond(atoms[current], atom)
            _add_bond(current, idx, order)
        pending_bond = None
        current = idx

    if stack:
        raise ValueError("括号未闭合：SMILES 末尾仍有未关闭的 (")
    if ring_open:
        raise ValueError(f"环闭合未配对：数字 {sorted(ring_open)} 只出现一次")
    if not atoms:
        raise ValueError("SMILES 中没有解析到任何原子")

    mol = Molecule(atoms=atoms, smiles=s)
    # 价态校验：显式 H 数超过价态即报错（如 [CH5]）。
    # 芳香原子按连接数计（吡咯型 [nH]：H 1 + 连接 2 = 3，合法）；非芳香按键级和计。
    for a in mol.atoms:
        if a.h_explicit is not None:
            if a.aromatic:
                if a.h_explicit + len(a.neighbors) > 3:
                    raise ValueError(
                        f"价态不合理：芳香 {a.element} 显式 H 数 {a.h_explicit} + 连接数 "
                        f"{len(a.neighbors)} 超过 3"
                    )
            else:
                valence = _VALENCE.get(a.element, 0)
                if a.element == "N" and a.charge > 0:
                    valence = 4
                bond_sum = sum(b for _, b in a.neighbors)
                if a.h_explicit + bond_sum > valence + 1e-9:
                    raise ValueError(
                        f"价态不合理：{a.element} 显式 H 数 {a.h_explicit} + 键级和 {bond_sum} 超过价态 {valence}"
                    )
        # 芳香 C 连接数超过 3 亦不合理（如 c(c)(c)c）
        if a.aromatic and a.element == "C" and len(a.neighbors) > 3:
            raise ValueError("价态不合理：芳香碳连接数超过 3")
    return mol


def describe(smiles: str) -> dict:
    """解析 SMILES 并返回全部描述符（JSON 可序列化）。"""
    return parse_smiles(smiles).as_dict()
