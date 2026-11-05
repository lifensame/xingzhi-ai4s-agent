# 部署指南（体验链接）

目标：拿到一个公开可访问的演示地址，填进技术文档的「应用体验链接」。
部署版默认 **offline 离线模式**，无需密钥即可完整体验（离线闭环 100/100）。

## 推荐：Streamlit Community Cloud（免费）

前提：一个 GitHub 账号 + 一个公开仓库。

1. 把 `xingzhi-ai4s-agent/` 的内容推到 GitHub 公开仓库，根目录须有：
   `app.py`、`requirements.txt`、`.streamlit/config.toml`、`data/`、`src/`、`ui/`。
   （发你的 zip 里就是这一套，直接解压上传即可；不要传 `.venv/`、`.env`。）
2. 打开 https://share.streamlit.io ，用 GitHub 登录 → **New app** →
   选仓库和分支，主文件填 `app.py` → **Deploy**。
3. 首次部署约几分钟，完成后得到形如 `https://xxx.streamlit.app` 的地址，
   就是体验链接。

注意：

- `.streamlit/config.toml` 会随仓库一起部署，深色主题自动生效，无需再配。
- `requirements.txt` 会装 openai/mcp（只在 live / MCP 时用到，离线不依赖），
  Cloud 上 pip 正常，不会复现本地 `versions: none` 的问题。
- 仓库更新后，在 Cloud 面板点 **Reboot** 即可重部署。

## 备选：Hugging Face Spaces（Streamlit SDK）

1. 打开 https://huggingface.co ，登录 → **New Space** → SDK 选 **Streamlit**，
   上传同样一套文件（或 `git push` 到 Space 的仓库）。
2. 得到 `https://huggingface.co/spaces/用户名/空间名`，就是体验链接。

注意：免费 CPU 档足够跑离线 demo；同样默认离线，无需密钥。

## Docker（仅供本地或自己的服务器）

```bash
docker build -t xingzhi-ai4s .
docker run --rm -p 8501:8501 xingzhi-ai4s ui
```

注意：Hugging Face Spaces 的 Docker 模式要求容器监听 **7860** 端口，
而当前 `docker-entrypoint.sh` 把 UI 写死在 8501，所以不建议走 Docker 模式上 HF；
真要走，先把 entrypoint 改成读 `$PORT` 环境变量。

## Live 模式与密钥

- 部署版默认 offline；**不要把密钥写进仓库或随 `.env` 提交**。
- 如需在部署版上开 live：把密钥配成平台的 Secret（Streamlit Cloud 的 Secrets /
  HF 的 Repository Secrets），再加一小段读取代码把 Secret 转成环境变量
  （当前版本只读本地 `.env`，不读平台 Secret）。需要的话找 Muse 加。
