<!-- generated_at: 2026-11-05T19:43:29.368989+08:00 Asia/Singapore -->
<!-- title: EGFR_VHH_文献扫描_理化校验与下一步实验设计 -->

# 实验笔记：EGFR VHH 文献扫描、理化校验与下一步实验设计

> 演示笔记 · 兴智杯 AI4S 科学智能体（搜—读—算—写—辩—赛）
> **声明**：本作品为参赛技术演示，与任何竞赛组委会或赛事主办方无隶属 / 赞助关系；示例序列为占位序列，不可用于真实湿实验。
> 本轮证据：[P001]、[P006]、[P037]、[P009]、[P016]（全部来自 Europe PMC 真实文献语料）

## 1. 意图

围绕 **EGFR** 靶点的 **VHH** 候选，在设计前完成：真实文献扫描 → 理化约束校验 → 下一步实验设计 → 可复现笔记沉淀。

## 2. 计划（控制环）

意图 → 计划 → literature_search → sequence_check（理化指标）→ design_experiment → 校验器（含引用接地）→ write_note。

## 3. 证据表（本轮检索命中）

| 引用 | 标题 | 年份 | 来源 |
|------|------|------|------|
| [P001] | Simultaneous solubility-enhancing domain removal and target purification using | 2027 | [10.1016/j.pep.2026.106988](https://doi.org/10.1016/j.pep.2026.106988) |
| [P006] | Validation and analysis of 12,000 AI-driven CAR-T designs in the Bits to Binde | 2026 | [10.1038/s44320-026-00246-1](https://doi.org/10.1038/s44320-026-00246-1) |
| [P037] | Improving de novo protein binder design with deep learning | 2023 | [10.1038/s41467-023-38328-5](https://doi.org/10.1038/s41467-023-38328-5) |
| [P009] | A modular nucleic acid-albumin biomolecular platform for multifunctional drug  | 2026 | [10.1016/j.jconrel.2026.115514](https://doi.org/10.1016/j.jconrel.2026.115514) |
| [P016] | De Novo Design of a Protein Binder to Probe Gas Channel and Enhance the Oxygen | 2026 | [10.1002/anie.6395554](https://doi.org/10.1002/anie.6395554) |

### 3.1 文献要点（每条主张均带引用）

- While protein A has become a reliable platform for purifying antibodies, an analogous universal purification method for native non-antibody therapeutics has yet to be widely implemented. [P001]
- Artificial intelligence (AI) methods for proteins have advanced rapidly, improving structure prediction and design, particularly for de novo binders. [P006]
- Recently it has become possible to de novo design high affinity protein binding proteins from target structural information alone. [P037]
- The nanomedicine field has seen a paradigm shift towards multifunctional drug designs to maximise drug efficacy. [P009]
- [NiFe]-Hydrogenases display remarkable catalytic efficiency for H2 production and oxidation but are sensitive to O2, which severely restricts their biotechnological applications. [P016]

## 4. 理化校验（pH 6.0 测定条件）

| 指标 | 数值 | 阈值 | 判定 | 依据 |
|------|------|------|------|------|
| 长度 length | 119 aa | 100–150 | ✅ | VHH 成熟肽常见 100–150 aa；de novo minibinder 常见 40– |
| 等电点 pI | 8.65 | 远离测定 pH 6.0±0.5 | ✅ | pI 落在测定缓冲 pH 附近时净电荷≈0，溶解度最低、易聚集 |
| 净电荷 @pH 6.0 | 2.06 e | |q| ≥ 1 | ✅ | 测定条件下保持一定净电荷有助于胶体稳定性 |
| 净电荷 @pH 7.4 | 1.69 e | 参考值 | — | 生理 pH 下的电荷参照 |
| GRAVY | -0.245 | ≤ 0 | ✅ | Kyte–Doolittle 平均疏水性；可溶球状蛋白通常为负 |
| 最大疏水补丁(9aa) | 1.167 | ≤ 1.8 | ✅ | 连续疏水窗口是聚集与非特异结合的主要代理指标 |
| 半胱氨酸 Cys | 2 个 | ≤ 4（经典 VHH 为 2） | ✅ | 游离 Cys 易致二聚与批次不稳；额外 Cys 需人工复核 |
| N-糖基化基序 N-X-S/T | 0 处 | 界面区应为 0 | ✅ | 真核表达体系中 N-X-S/T 可被糖基化，可能遮蔽结合界面 |
| CDR3 长度(估计) | 10 aa | 8–24 | ✅ | 骆驼科天然库 CDR3 偏长，过短难覆盖凹陷表位、过长熵损失大 |
| 分子量 MW | 12891.2 Da | — | — | 用于纯化与质谱确认 |
| ε280 | 31525 M⁻¹cm⁻¹ | — | — | A280 定量浓度所需 |
| 芳香族比例 | 0.143 | — | — | 影响 A280 与界面堆积 |
| 疏水残基比例 | 0.336 | — | — | 整体疏水负荷参考 |

序列指纹 `sha256:3639767f2b7ad323`，长度 119 aa，总体判定：通过 [TOOL:sequence_check]。

### 4.1 风险提示

- 无额外风险提示。

## 5. 下一步实验设计（按判别价值排序）

**优先级 1｜Poly-specificity reagent (PSR) 流式打分 + 基线 BLI 非特异结合（n=2 重复）** [TOOL:design_experiment]

- 触发风险：最大疏水窗口 1.167，非特异结合与聚集风险 [TOOL:sequence_check]
- 对照：高非特异对照抗体 + 低非特异临床抗体各一
- 读出：PSR 归一化 MFI；BLI 对空白传感器的漂移 [TOOL:design_experiment]
- 判别价值：中高：PSR 分数 >0.3 则优先做疏水补丁回突变，而不是继续优化亲和力 [TOOL:design_experiment]
- 成本：低（2 天）

**Go / No-Go** [TOOL:design_experiment]：进入下一轮的判据：pH 6.0 下 SEC 单体率 ≥ 90%、PSR 非特异分数 < 0.3、BLI KD < 100 nM（对 EGFR），任一不满足则回到序列层面修复而非扩大筛选规模。

每条件 n≥3 生物学重复；先做最便宜且判别力最高的那一项（队列第 1 项）。 [TOOL:design_experiment]

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