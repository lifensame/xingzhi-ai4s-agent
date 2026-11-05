"""UI 主题与样式 / Dark theme CSS for the Streamlit demo."""

ACCENT = "#6EE7B7"
ACCENT_2 = "#6EE7B7"

CSS = """
<style>
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],#MainMenu,.stDeployButton{display:none !important;}
:root {
  --accent: #6EE7B7;
  --accent2: #6EE7B7;
  --bg: #0B0F14;
  --card: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.09);
  --muted: #8FA3B8;
}
html, body, [class*="css"] { font-feature-settings: "tnum"; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

.xz-hero {
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 26px 30px;
  background:
    radial-gradient(900px 300px at 12% -20%, rgba(110,231,183,.18), transparent 60%),
    radial-gradient(700px 260px at 88% -10%, rgba(110,231,183,.08), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
  box-shadow: 0 18px 50px rgba(0,0,0,.45);
  margin-bottom: 20px;
}
.xz-hero h1 {
  margin: 0 0 6px 0; font-size: 2.05rem; letter-spacing: .4px; font-weight: 750; line-height: 1.3;
  background: linear-gradient(92deg, #FFFFFF 10%, var(--accent) 55%, var(--accent2) 95%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.xz-sub { color: var(--muted); font-size: .96rem; line-height: 1.7; overflow-wrap: anywhere; }
.xz-chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.xz-chip {
  border: 1px solid var(--stroke); border-radius: 999px; padding: 4px 13px;
  font-size: .78rem; color: #CBD5E1; background: rgba(255,255,255,.04);
}
.xz-chip.on { color: #04150F; background: var(--accent); border-color: transparent; font-weight: 650; }

.xz-card {
  border: 1px solid var(--stroke); border-radius: 16px; padding: 16px 18px;
  background: var(--card); margin-bottom: 12px;
  transition: border-color .2s ease, transform .2s ease;
}
.xz-card:hover { border-color: rgba(110,231,183,.45); transform: translateY(-1px); }
.xz-card .t { font-weight: 680; font-size: .95rem; margin-bottom: 4px; overflow-wrap: anywhere; word-break: break-word; }
.xz-card .d { color: var(--muted); font-size: .86rem; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }

.xz-step { display: flex; gap: 14px; align-items: flex-start; min-width: 0; }
.xz-step > div { min-width: 0; }
.xz-dot {
  width: 30px; height: 30px; min-width: 30px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; font-size: .9rem;
  border: 1px solid var(--stroke); background: rgba(255,255,255,.05);
}
.xz-dot.ok { background: rgba(110,231,183,.16); border-color: rgba(110,231,183,.5); }
.xz-dot.err { background: rgba(248,113,113,.16); border-color: rgba(248,113,113,.5); }
.xz-dot.run { animation: pulse 1.4s ease-in-out infinite; }

.xz-kpi { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 14px; }
.xz-kpi .k {
  flex: 1 1 110px; border: 1px solid var(--stroke); border-radius: 14px;
  padding: 10px 12px; background: var(--card);
}
.xz-kpi .k .lab { color: var(--muted); font-size: .74rem; letter-spacing: .6px; text-transform: uppercase; }
.xz-kpi .k .val { font-size: 1.2rem; font-weight: 720; margin-top: 3px; }
.good { color: var(--accent); } .bad { color: #F87171; } .warn { color: #FBBF24; }

.xz-note {
  border: 1px solid var(--stroke); border-radius: 16px; padding: 10px 22px 18px;
  background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.012));
  overflow-wrap: anywhere; word-break: break-word; min-width: 0;
}
.xz-note h1 { font-size: 1.5rem; } .xz-note h2 { font-size: 1.12rem; margin-top: 1.1rem; }
/* 笔记里的 sha256 哈希、长 URL 等无断点长串必须能断行，否则会撑破窄列。
   注意：不能用 <div class="xz-note"> 包住笔记——Streamlit 遇到原始 HTML 块后
   不再解析其中的 Markdown，会把 ### 标题原样显示为文字；因此只针对行内 code
   （段落/列表/表格里的 code），围栏代码块（pre > code）保持横向滚动。 */
[data-testid="stMarkdownContainer"] p code,
[data-testid="stMarkdownContainer"] li code,
[data-testid="stMarkdownContainer"] td code { word-break: break-all; overflow-wrap: anywhere; }
[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
.xz-foot { color: var(--muted); font-size: .78rem; margin-top: 18px; line-height: 1.6; }

div.stButton > button[kind="primary"] {
  background: linear-gradient(92deg, var(--accent), var(--accent2));
  color: #04150F; border: 0; font-weight: 700; border-radius: 12px; height: 46px;
  transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 10px 26px rgba(110,231,183,.28);
}
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(110,231,183,.45);} 50% { box-shadow: 0 0 0 9px rgba(110,231,183,0);} }

.xz-cite {
  display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: .3px;
  padding: 1px 7px; margin: 0 2px; border-radius: 999px; vertical-align: 1px;
  color: #6EE7B7; background: rgba(110,231,183,.10); border: 1px solid rgba(110,231,183,.32);
  white-space: nowrap; /* 徽章不在窄列里从中间断开（P001 曾被断成 P/00/1 三行） */
}
.xz-cite.tool { color: #CBD5E1; background: rgba(148,163,184,.10); border-color: rgba(148,163,184,.32); }
.xz-note table { font-size: .86rem; }
/* Markdown 表格（笔记里的证据表/理化校验表）在窄列中改为容器内横向滚动，
   不再把表格宽度（证据表 176px、理化表 210px）顶出 151px 宽的笔记列。 */
[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }
/* 表格单元格不从词中任意断开：年份 2027 曾被断成 20/27；DOI 链接仍可在 / 与 . 后自然换行。 */
[data-testid="stMarkdownContainer"] td { overflow-wrap: normal; }
/* 窄屏时分栏上下堆叠：每列最小 320px，容器放不下两列就换行（时间线/笔记栏在 900px
   时各只剩约 180px、文字逐词换行；堆叠后各占整宽）。对反例演示的两栏同样适用。 */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: min(320px, 100%); }
</style>
"""

CSS = CSS.replace("</style>", """
h4 { margin-top: 1.6rem !important; font-weight: 680; letter-spacing: .2px; }
.xz-empty {
  border: 1px dashed var(--stroke); border-radius: 16px; padding: 26px 20px; text-align: center;
  background: rgba(255,255,255,.015); margin-bottom: 12px;
}
.xz-empty .i { font-size: 1.6rem; opacity: .7; }
.xz-empty .t { font-weight: 650; margin-top: 6px; }
.xz-empty .d { color: var(--muted); font-size: .85rem; margin-top: 4px; line-height: 1.6; }
.xz-metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap: 10px; margin-bottom: 12px; }
.xz-metric { border: 1px solid var(--stroke); border-radius: 14px; padding: 12px 14px; background: var(--card); }
.xz-metric .lab, .xz-kpi .k .lab { color: var(--muted); font-size: .74rem; letter-spacing: .4px; }
.xz-metric .val { font-size: 1.25rem; font-weight: 720; margin-top: 4px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; word-break: break-word; }
.xz-metric .u { font-size: .72rem; color: var(--muted); margin-left: 4px; font-weight: 500; }
.xz-metric .th { color: var(--muted); font-size: .72rem; margin-top: 4px; overflow-wrap: anywhere; }
.xz-kpi .k .val { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>""")
