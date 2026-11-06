# 兴智杯 · AI4S 科学智能体（搜—读—算—写—辩—赛）

第三届「兴智杯」大模型智能体关键能力技术创新赛 · **方向1：AI4S 科学智能体** 参赛演示包。

> **一句话**：它不只写带引用的科研笔记，还告诉你**哪些结论站得住、文献为什么打架、下一步该做哪个实验来裁决**——由两项新能力实现：**证据辩证**（`evidence.py`）与**假设锦标赛**（`hypotheses.py`）。
> 基座对接**书生大模型**（InternLM）OpenAI 兼容接口，已实测跑通 LIVE 闭环。
> **默认 OFFLINE**：无 API Key 也能一键跑通 Demo，确定性复现并产出 Markdown 实验笔记。
> 演示叙事围绕 EGFR / VHH 文献扫描与设计约束（声明见 §安全与合规）。
> **语料是真的**：52 篇 Europe PMC 真实文献（全部带 DOI/PMID，可点击溯源），由 `scripts/fetch_corpus.py` 可随时重新抓取。

---

## 1. 60 秒上手

环境要求：**Python ≥ 3.12**（与 `Dockerfile` 的 `python:3.12-slim` 一致；`pandas` 与 `streamlit` 均要求 ≥ 3.11）。主流程不依赖 `make`。

### 一键打开（推荐）

解压后双击即可，不用手动建虚拟环境、装依赖：

- **Windows**：双击 `start.bat`
- **macOS**：双击 `start.command`
- **Linux**：终端运行 `./start.sh`

首次运行会自动创建 `.venv`、安装依赖，然后打开浏览器进入界面（http://localhost:8501）。再次双击会直接启动，不会重复安装。已在全新目录实测：从零开始可完整跑通并打开界面。

说明：启动脚本先装核心依赖（界面与离线演示必需），再装扩展依赖 `openai`（仅 LIVE 模式用）与 `mcp`（仅 MCP 服务用）。如果你的网络或镜像源没有 `openai==3.45.0` 这个版本，脚本会自动换源重试，实在装不上也会跳过并继续启动——离线演示与界面功能不受影响，只有 LIVE 模式与 MCP 服务不可用。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # 含运行 + 测试依赖

# ① 离线闭环（零配置，评委可直接复现）
python scripts/run_demo.py --mode offline

# ② 可视化控制台（推荐演示 / 录屏）
python scripts/run_ui.py                # 等价于 streamlit run app.py
# 浏览器打开 http://localhost:8501

# ③ LIVE（书生 API，可选）
cp .env.example .env                    # 填入 INTERNLM_API_KEY
python scripts/run_demo.py --mode live

# ④ 测试 + 一键验证
python -m pytest -q
bash scripts/verify_all.sh

# ⑤ 重新抓取真实语料（Europe PMC，无需任何 Key）
python scripts/fetch_corpus.py
```

> 可选：装有 `make` 的机器也可用 `Makefile` 中的快捷目标（如 `make docker`），但非必需。
> 注：Docker 镜像未在本评审环境构建验证（本机无 Docker）；提交前请在有 Docker 的机器上跑一次 `docker build` + `docker run` 验证。

---

## 1.5 创新点（主线 2 项 + 底座 5 项，均可现场演示）

### 主线：证据辩证 + 假设锦标赛（新）

| # | 能力 | 它做什么 | 怎么看到 | 实测（最新离线运行 `outputs/last_run_manifest.json`） |
|---|------|----------|----------|------|
| A | **证据辩证 / evidence adjudication**（`src/xingzhi_agent/evidence.py`） | 把笔记每句的引用支撑分成 **直接 / 间接 / 过度表述 / 无支撑** 四级；"证明""必然"等过度表述自动降级为稳健措辞；找出同主题结论相反的文献，区分**实质分歧**与**条件不同 / 指标口径不同**造成的伪冲突；给每个主题 0–100 置信度。内置 48 个术语的中英术语桥，让中文笔记能与英文摘要比对。全部确定性、不调用模型。 | UI「⚖ 证据辩证」面板（`docs/screenshots/ui_evidence.png`、`ui_disagreement.png`） | 直接 11 / 间接 0 / 过度表述 0 / 无支撑 0，支撑分 100；语料中实质分歧 3 处（`cdr3_repertoire` 2 处、`pH_dependent_binders` 1 处），该主题置信度 30（8 个主题 20–70，最低为 pH_dependent_binders 20） |
| B | **假设锦标赛 / hypothesis tournament**（`src/xingzhi_agent/hypotheses.py`） | 由分歧与缺口生成 **6 个可检验假设**（冲突按 topic×claim_direction 去重）→ 红队**相对主张方向**计票 → **15 场循环赛 + Elo**（优先 contest）→ 存活分 → 冠军的**最小判别性下一步实验**。 | UI「🏆 假设锦标赛」面板（`docs/screenshots/ui_tournament.png`） | 6 假设（2 冲突+4 缺口）、15 场；冠军 **H02，Elo 1054.5，存活分 54，contest 1**；support=[P007]，counter=[P024] |

> 诚实说明：两项能力都基于**词面重叠 + 关键词规则**，术语桥为手工构建；手写样例集 `tests/fixtures/support_cases.json` 上支撑分级 10/10、分歧判型 6/6，但样例只有 10 + 6 条、由我们自己编写，不代表真实准确率；锦标赛裁判是规则化的确定性裁判（H02 因 contest=1 且证据净值高于 H01 胜出，不是"科学正确"）。红队**已按** `claim_direction` 相对计票（H02 support=[P007]，H01 support=[P029]）。详见 `docs/tech_report.md` 第 4.1 / 4.8 / 6 节与 `docs/scoring_alignment.md`。
>
> 新模块与红队修复后：`pytest -q` **192 passed**；`scripts/verify_all.sh` **PASS=6 FAIL=0**；离线 100/100，笔记哈希为 `80ce2e27…`（副标题改为六阶段后；改动前为 `dcda7315…`）；一次 LIVE 运行（intern-s2）**100 分、45.2 s**（`outputs/runs/20261105-105718-live-cc4441c2.json`，该次覆盖证据辩证，早于锦标赛接入）。

### 底座（让主线可信的地基）


| # | 创新点 | 它解决什么 | 怎么看到 | 实测 |
|---|--------|-----------|----------|------|
| ① | **可验证引用闭环 / claim-level grounding**（`src/xingzhi_agent/grounding.py`） | LLM 科研写作最大的问题是"看似有引用、其实对不上"。我们把笔记切成句级主张，逐条做"主张 ↔ 证据"词元重叠匹配；文献用 `[Pxxx]`，工具算出的数值用 `[TOOL:sequence_check]` / `[TOOL:design_experiment]`，且该工具必须本轮真跑过才算数。接地率进校验总分（50%）。 | UI 笔记里引用被渲染成绿色/蓝色徽章；KPI 有「引用接地」百分比 | 离线接地 **100%**；LIVE 首轮 42% → 被确定性引用对齐修正 |
| ② | **自修复环 / critic-repair loop**（`agent.py::_finish`） | 校验不只是"打分并放行"，而是把校验器反馈回灌给模型重写，最多 3 轮；无提升则回退上一版并停止，不会越改越差。 | UI 的「得分轨迹」折线图 + 时间线上的 🩹 修复步 | 离线 100/100 → 0 轮；LIVE 见 §6 |
| ③ | **一级原理理化指标**（`physchem.py`，零重依赖，纯标准库） | pI（Bjellqvist pK + 二分法）、任意 pH 净电荷（Henderson–Hasselbalch）、GRAVY（Kyte–Doolittle）、分子量、ε280（Pace）、最大疏水窗口、Cys 计数与奇偶、N-X-S/T 糖基化位点、CDR3 长度（IMGT 近似）。每项都带阈值与依据，不是"模型随口说的数"。 | UI「理化指标」卡片，10 张卡各带阈值判定 | 对 119 aa VHH：MW 12891.2 Da、pI 8.65、pH6.0 净电荷 2.06、GRAVY −0.245、CDR3 10 aa；单测对照 GG 二肽 132.1 Da、全 Ile GRAVY 4.5、pI 处净电荷≈0 |
| ④ | **实验设计建议器**（`experiment.py`） | 输出的不是"建议做进一步实验"，而是按**判别价值**排序的具体方案：触发风险 → 实验 → 对照 → 读出 → 判别价值 → 成本，外加 Go/No-Go 判据与重复数。风险项由理化数据触发（pI 贴近实验 pH、疏水窗口、Cys 奇偶、糖基化位点）。 | 笔记第 5 节 + UI 时间线「设」阶段 | 如 pI 6.41 对 pH 7.4 → 自动给出"pH 7.4 vs 8.4 SEC-HPLC 单体率对比，n=3"，并说明差 >15% 即判定电荷驱动聚集 |
| ⑤ | **成本 / 可复现账本**（`ledger.py`） | 每轮记录 LLM 调用数、token、各段延迟、输入/输出 sha256、语料 sha256、seed 策略，落盘 `outputs/last_run_manifest.json`。离线模式相同输入必得相同 `note_sha256`（已写进 `scripts/verify_all.sh` 自动校验）。 | UI 底部「成本/可复现账本」面板 | 离线 `note_sha256` 两次运行一致 `80ce2e27cd71459e…`；语料 sha256 `ac30f79c68f83222…`（P2-1 补图表元数据后；原 `6d58293f…` 见 `outputs/_archive/corpus.json.bak`） |

---

## 2. 赛题对齐

### 2.1 五个考察点 → 本作品能力

| 赛题考察点 | 本作品能力 | 现场怎么看 |
|------------|------------|------------|
| **自主规划** | 搜—读—算—写—辩—赛控制环 | UI 时间线逐步点亮 |
| **工具调用** | 文献检索 + 理化计算（失败可重试） | 蓝徽章 `[TOOL:…]`、工具轨迹 |
| **多步骤推理** | 自修复 + 证据辩证 + 假设锦标赛 | 得分轨迹；⚖ / 🏆 面板 |
| **科学计算** | 一级原理 pI / GRAVY / 净电荷等可重算 | 理化指标卡片 |
| **预测泛化** | 离线确定性复现 + LIVE 书生 API；8 主题子集不改 prompt 均可产出假设 | `note_sha256=80ce2e27…`；`docs/ablation_generalization.md` |

### 2.2 提交与闭环要点

| 项 | 本作品 |
|----|--------|
| 方向 | 方向1 · AI4S 科学智能体 |
| 闭环 | **搜** → **读** → **算** → **写** → **辩** → **赛** |
| 控制环 | 意图 → 计划 → 工具调用 → 校验器 → 辩证/锦标赛 → 笔记沉淀，每步事件化、可回放 |
| 书生绑定 | `INTERNLM_BASE_URL` + `INTERNLM_API_KEY`，OpenAI 兼容 `chat/completions` |
| 可信性 | 校验器做合规表述 / 引用可追溯 / 结构完整 / 密钥扫描四类检查并打分 |
| 可复现 | 离线确定性脚本 LLM；语料、规则、产物全部落盘在仓库内 |
| 可落地 | 一键 CLI + 暗色可视化控制台 + `outputs/*.md` 产物可下载 |
| 提交物 | 技术报告 + 可复现代码/环境 + 2–3 min 演示；Zip：单位简称-作品名称-队长姓名-队长手机号-日期.zip |

**非目标（刻意不做）**：真实湿实验预约、联网全文爬取、结构预测重计算、官方竞赛代报名。

### 2.3 方向对齐加分项（P2：多模态 / MCP / 多智能体）

| 项 | 最小可信实现 | 明确不做的事 |
|----|--------------|--------------|
| **多模态图表解析**（`figures.py` + `parse_figure` 工具） | 语料 52 篇中 **37 篇**有真实图表元数据（Europe PMC fullTextXML `<fig>`，共 **237 张图**，见 `data/corpus.json` 的 `figures` 字段与 `provenance.figures_fetch`）；图注做规则数值抽取（数值 + 单位 + pH/温度等条件），图表主张用 `[Pxxx:FigN]` 引用、走同一套接地判定（图注即证据文本） | 不做图像像素 OCR、不做视觉模型读图；其余 15 篇为非开放获取、无全文，如实记 `figures: []`，不编造 |
| **MCP 封装**（`src/xingzhi_agent/mcp_server.py`，stdio） | 官方 MCP Python SDK v2（`mcp==2.3.0`；v2 把 FastMCP 更名为 MCPServer）薄封装六个工具：`literature_search` / `sequence_check` / `design_experiment`（P0–P2）与 P3 新增的 `fit_binding_curve` / `design_protocol` / `analyze_smiles`，不重写工具逻辑；`python scripts/run_mcp_server.py` 启动，实测 `list_tools` 返回 6 个工具、`sequence_check` 经 MCP 的返回值与直接调用逐字段一致 | 离线 demo 不依赖 MCP（卸载 `mcp` 后仍可跑） |
| **多智能体分工**（`roles.py`，`run_demo.py --team`，默认关闭） | 检索员 / 计算员 / 评审员三角色（提示词 + 工具子集 + 事件标签），步骤与账本打角色标签，UI 时间线按角色展示，账本 `outputs.roles` 汇总各角色工具调用数 | **单模型多角色编排**，不是多个独立智能体；不新增 LLM 调用；不引入 LangGraph/CrewAI 等重型框架 |
| **P3 新工具：实验曲线拟合 / 实验方案 / 分子结构解析**（`curves.py` / `protocol.py` / `smiles.py`，纯标准库） | 4 参数 logistic 拟合结合曲线（EC50、Hill、R² + 质量标记，粘贴 CSV 可跑；合成曲线已知 EC50=100 nM 回收 100.00 nM、R²=1.0000）；C1V1=C2V2 稀释方案生成器（梯度、对照、读出、Go/No-Go；BLI 样例最高点 2.0 µL + 198.0 µL 缓冲液）；SMILES 限定子集解析（分子式/分子量/HBD/HBA/可旋转键/TPSA/Lipinski；咖啡因 C8H10N4O2、MW 194.19，TPSA 估算 66.4 Å² vs PubChem 公布 61.82 Å² 已注明偏差原因）；三工具均已进 MCP（共 6 个）与 UI 折叠面板，`scripts/run_p3_demo.py` 可复现（产物 `outputs/sample_p3.md`） | 默认离线闭环不调用新工具（笔记哈希仍为 `80ce2e27…`）；SMILES 不含立体化学与 % 环号 >9、有机金属除外；TPSA 与 logP 为估算值，不用于合规判定 |

MCP 客户端配置示例（Claude Desktop / 任意 MCP 客户端）：

```json
{"mcpServers": {"xingzhi-ai4s": {"command": "python",
  "args": ["scripts/run_mcp_server.py"], "cwd": "<包根>"}}}
```

重新抓取图表元数据：`python scripts/fetch_corpus.py --figures-only`（只补 `figures` 字段，不重抓文本；需联网，离线运行时不调用）。

---

## 3. 架构

```
                 用户科研目标（中文自然语言）
                            │
                            ▼
        ┌──────────── ScienceAgent（agent.py）────────────┐
        │  LLM 后端：OfflineLLM（确定性） | LiveLLM（书生）  │
        │  控制环：意图 → 计划 → 工具 → 校验 → 写作         │
        │  事件总线 on_event ──────────────► Streamlit UI  │
        └───────┬──────────────┬──────────────┬───────────┘
                │              │              │
     literature_search   sequence_check    write_note
       data/corpus.json   100–150aa(VHH)   outputs/*.md
       52 篇真实文献      40–100aa(mini)         │
                │              │                │
                └──────────────┴────────┬───────┘
                                        ▼
                             verifier.py 校验器打分
                      合规 / 引用可追溯 / 结构 / 密钥扫描
                                        │
                                        ▼
                    evidence.py 证据辩证（支撑分级 / 降级 / 分歧 / 置信度）
                                        │
                                        ▼
                 hypotheses.py 假设锦标赛（生成 / 红队 / Elo / 下一步实验）
                                        │
                                        ▼
                         ledger.py 运行账本 last_run_manifest.json
```

目录：

```
app.py                 可视化控制台（Streamlit，暗色主题）
ui/theme.py            UI 主题与 CSS
scripts/run_demo.py    CLI 一键 Demo（中文输出）
scripts/run_ui.py      一键启动 UI
src/xingzhi_agent/     agent / llm / tools / verifier / grounding / repair / evidence / hypotheses / ledger / physchem / experiment / config
data/corpus.json       52 篇 Europe PMC 真实文献语料
outputs/               实验笔记产物（含 sample_lab_note.md / sample_live_lab_note.md）
tests/                 192 项测试，离线全绿（含 fixtures/support_cases.json 手写样例）
```

> **outputs/ 冻结快照**：截至 2026-11-05，顶层只保留 `latest_lab_note.md`、`sample_lab_note.md`、`sample_live_lab_note.md`、`sample_live_lab_note_100.md`、`SELF_REVIEW.md`、`last_run_manifest.json`、`ablation_generalization.json`、`baseline/`、`sample_runs/`；`runs/` 只保留 10 个代表性账本（022520-live、105718-live、114329-offline、150425-offline、162050、162054、184807、184958、185010、185014，离线笔记哈希 `80ce2e27…`）；其余历史产物见 `outputs/_archive/README.md`；保留账本内的路径均为相对路径。

---

## 4. 评委会看到什么

1. **一条命令跑通闭环**：终端逐步打印 意图 → 计划 → 工具调用 → 工具结果 → 校验器得分 → 笔记路径。
2. **可视化控制台**：暗色界面，左侧时间线逐条动画出现（搜/读/算/写/辩/赛 阶段徽章点亮），右侧实时渲染实验笔记，顶部 KPI 卡展示 闭环状态 / 校验得分 / 工具调用数 / 用时 / 模式。
3. **证据辩证 + 假设锦标赛**：笔记下方两块面板，逐句支撑强度、文献分歧判型、主题置信度、6 个假设的 Elo 排名与冠军假设的下一步判别实验。
4. **可审计**：展开「校验器明细」看每项检查通过与否；展开「工具调用轨迹」看每步入参与原始返回 JSON。
5. **双模式**：侧边栏一键在 offline（零配置复现）与 live（书生 API）之间切换。
6. **产物**：`outputs/latest_lab_note.md` 可一键下载。

---

## 5. 录屏脚本建议（2–3 分钟）

| 时间 | 画面 | 旁白要点 |
|------|------|----------|
| 0:00–0:30 | 赛题考察点表 + 架构图 | 「五考察点对上：规划/工具/多步推理/科学计算/泛化；闭环是搜—读—算—写—辩—赛。」 |
| 0:20–0:50 | 终端 `python scripts/run_demo.py --mode offline` | 「零配置，不需要任何密钥，确定性复现——评委拿到就能跑。」 |
| 0:50–1:10 | UI 点击「启动闭环」 | 「左侧是控制环时间线；笔记里绿色徽章是文献、蓝色是工具算出的数值。」 |
| 1:10–1:45 | 「⚖ 证据辩证」面板 | 「每句话的支撑强度四级分级；CDR3 主题检出 2 处实质分歧，置信度 30（8 个主题 20–70，最低为 pH_dependent_binders 20）。」 |
| 1:45–2:15 | 「🏆 假设锦标赛」面板 | 「6 个假设、15 场对决，冠军 H02（Elo 1054.5，存活分 54，support=[P007]），以及给它的最小判别实验。」 |
| 2:15–2:45 | 运行账本 / 运行档案 | 「全部写进账本；书生 intern-s2 的真实运行 100 分、45.2 秒。」 |
| 2:45–3:00 | 局限 | 「词面匹配 + 规则裁判，样例小且自写——如实说明。」 |

---

## 6. LIVE（书生）配置

```ini
DEMO_MODE=live
INTERNLM_API_KEY=<你的 Key，只放在本地 .env>
INTERNLM_BASE_URL=https://chat.intern-ai.org.cn/api/v1
INTERNLM_MODEL=intern-s2
INTERNLM_TIMEOUT=120    # 主循环单步上限；实测最慢一步（写整篇笔记）32.5s，120s 是约 3.7 倍余量
INTERNLM_RETRIES=2
```

实测记录（2026-11-05 Asia/Singapore）：

| 配置 | 结果 |
|------|------|
| `chat.intern-ai.org.cn/api/v1` + `intern-s2` | ✅ 单轮问答 ~6s |
| 同上，四工具完整闭环（2026-11-05 10:57 Asia/Singapore，新模块后） | ✅ **100/100，45.2s，4 次 LLM 调用，15,337 tokens** |
| 同上 + `intern-latest` | ❌ 90s 超时（该账号下不可用/排队；无账本，口头记录） |

新模块后 LIVE 主证据：`outputs/runs/20261105-105718-live-cc4441c2.json`（100 分、45.2 秒、4 次调用、15,337 tokens；接地 54% → 100%，修复 0 轮；支撑直接 20）。优化前对照见 `outputs/runs/20261105-022520-live-8fc48102.json`（81 分、2190.33 秒、13 次调用、155,020 tokens；轨迹 70 → 70 → 81，第 3 轮因连接错误失败、分数为 None）。

> 产物见 `outputs/sample_live_lab_note.md`，账本见 `outputs/sample_live_manifest.json`。
> 早一版（`INTERNLM_TIMEOUT=150`）的修复调用会在重写长笔记时超时并回退到 66/100——那时修复是整篇重写；现版修复只重写一个未通过的小节（≤220 字），单轮上限 90s、总预算 180s，不再需要 300s。主循环单步上限现为 120s（实测最慢一步写整篇笔记 32.5s，约 3.7 倍余量），超时会明确报错并提示切回 offline，不再静默卡住。

> `intern-s2` 为推理型模型，单步较慢，`INTERNLM_TIMEOUT` 默认 120s；演示现场建议用 offline，LIVE 作为「可切换真实基座」的证明。
> LIVE 失败时，智能体会打印中文错误并建议回落 offline；UI 也会给出黄色提示自动回落。

---

## 7. 安全与合规

- 密钥只存在本地 `.env`（已 gitignore），仓库仅含 `.env.example`；异常信息经正则脱敏，不会外泄 Key。
- 校验器扫描笔记中的疑似密钥，并禁用「官方指定 / 主办方认证 / 保证获奖 / 真实湿实验已测」等表述。
- 语料为 **Europe PMC 真实文献**（52 篇，1994–2027，全部带 DOI/PMID 与原文链接），内部 `id` 形如 `P001`；笔记只能引用本轮检索命中的 ID，引用外 ID 会被校验器标记。`data/corpus.bundled.json` 为离线兜底副本，`data/corpus.json` 的 `provenance` 字段记录抓取时间、查询词与 sha256。
- 本作品为参赛技术演示，与任何竞赛组委会或蛋白质设计赛事**无隶属、无赞助、无认证关系**；序列仅为示例，不可用于真实湿实验或临床用途。
- 本作品**不提供医疗/临床建议**；实验设计建议仅供演示评审参考，不可作为真实实验或临床决策依据。
- 语料中的标题与摘要片段版权归原出版方所有，仅用于参赛演示中的引用与溯源展示。

---

## 8. 提交前检查清单

- [ ] `pytest -q` 全绿（**192 项**）
- [ ] `bash scripts/verify_all.sh` 输出 `ALL PASS`（PASS=6 FAIL=0）
- [ ] `python scripts/run_demo.py --mode offline` 成功写出笔记，校验 100/100
- [ ] `python scripts/run_ui.py` 可打开，点击「启动闭环」时间线正常
- [ ] `.env` / `.token_raw` / `.venv` / `__pycache__` **未**进入提交包
- [ ] 录屏 2–3 分钟，含离线复现 + UI + 证据辩证面板 + 假设锦标赛面板
- [ ] README 架构图与实际目录一致
- [ ] 全包检索无密钥：`grep -rnE "sk-[A-Za-z0-9]{16,}|eyJ[A-Za-z0-9._-]{24,}" --exclude-dir=.venv --exclude-dir=.git --exclude=".env" --exclude=".token_raw" --exclude-dir=tests .`（与 `scripts/verify_all.sh` 第 5 项同一正则；排除 `tests/` 是因为 `test_verifier_flags_secret` 需用一枚密钥形态的假字符串来测试校验器本身）
- [ ] 自评报告 `outputs/SELF_REVIEW.md` 已更新
- [ ] 最终 Zip 按模板重命名：单位简称-作品名称-队长姓名-队长手机号-日期.zip，并核对队长姓名 / 手机号 / 日期

---

## 9. 界面截图

暗色主题、单一强调色（#6EE7B7）。截图由 `streamlit run app.py` + 无头 Chrome（1440 宽）实拍；账本与运行档案截图读取的是**真实书生 API（intern-s2）运行**的存档（`outputs/runs/*-live-*.json`），不是离线占位数据。

**实测结果（2026-11-05）**

| 项 | 结果 |
|----|------|
| 离线模式 | 100/100，两次运行笔记 sha256 一致（可复现） |
| 真实 API（intern-s2）连跑 4 次 | 95 / 95 / 96 / 100 分，每次 87–105 秒，均一轮达标无需重写 |
| 引用接地率 | 原始 34–59% → 经确定性引用对齐后 91–100% |
| 早期对照运行 | 自修复环 70 → 70 → 81（耗时 2190 秒；第 3 轮因连接错误失败、分数为 None）——这正是后来加入引用对齐与重写时间预算的原因 |
| 测试 | `pytest -q` 192 passed |
| 证据辩证（离线） | 直接支持 11 / 过度表述 0 / 无支撑 0；语料实质分歧 3 处；支撑分级样例 10/10、分歧判型样例 6/6（自写小样例） |
| 假设锦标赛（离线） | 6 个假设（2 冲突+4 缺口）· 15 场 · 冠军 H02（Elo 1054.5 · 存活分 54 · contest 1 · support=[P007] · counter=[P024]） |
| 新模块后 LIVE | 100 分，45.2 秒（`outputs/runs/20261105-105718-live-cc4441c2.json`） |
| 消融（决策就绪分，报告指标非官方分） | full 100 → no_tournament 85（−15）→ no_evidence 50（−50）→ no_physchem 79.4（−20.6）；no_repair 100（round-0 已达标） |
| 跨主题泛化 | 8 个 ≥3 篇主题均产出假设+next_experiment；ESMFold_structure、cdr3_repertoire 与 EGFR_nanobody 不相交 |
| 已修复 | 模型把数值参数写成字符串（如 `"pH5.0"`）导致流程崩溃：现已容错解析，工具报错回喂模型重试 |

| 证据辩证（逐句支撑强度） | 文献分歧判型 + 主题置信度 |
|----------|------------------------|
| ![证据辩证](docs/screenshots/ui_evidence.png) | ![分歧与置信度](docs/screenshots/ui_disagreement.png) |

| 假设锦标赛（Elo 排名 + 下一步判别实验） | 团队分工（单模型多角色编排） |
|----------|---|
| ![假设锦标赛](docs/screenshots/ui_tournament.png) | ![团队分工](docs/screenshots/ui_team.png) |

| 首屏（未运行时的空状态 + 最近存档轨迹） | 主控制台（KPI + 控制环时间线 + 笔记渲染） |
|----------|------------------------|
| ![首屏](docs/screenshots/ui_home.png) | ![主界面](docs/screenshots/ui_main.png) |

| 质量分折线（各修复轮次 · 90 分目标线） | 理化指标卡片 + 证据面板（DOI 可点击） |
|----------|------------------------|
| ![评分轨迹](docs/screenshots/ui_trajectory.png) | ![指标与证据](docs/screenshots/ui_metrics_evidence.png) |

| 运行账本（模型 · 逐步延迟 · token · seed · 停止原因） | 运行档案（outputs/runs/ 历史） |
|----------|------------------------|
| ![账本](docs/screenshots/ui_ledger.png) | ![档案](docs/screenshots/ui_run_history.png) |

| 反例演示（校验器拦截） | |
|----------|---|
| ![反例](docs/screenshots/ui_guardrails.png) | |
