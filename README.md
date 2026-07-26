# Mistake Notebook（错题整理）

Mistake Notebook 是一个本地优先的错题照片整理工具。它可以校正拍照透视和纸面弯曲、识别印刷题干与 LaTeX 公式、保留原题配图、清除手写区域，并按“一级领域 / 二级题型”分类，最终导出可编辑 DOCX 或 Obsidian Markdown。

系统以题意忠实性为第一原则：公式只有在能与原题文字可靠对齐时才替换；无法确认的公式或配图保留原图证据并要求人工复核，不根据题意猜测内容。

## 支持的平台

- macOS 12 及以上：默认使用 macOS Vision；Apple Silicon 可使用 PyTorch MPS，PaddleOCR 使用 CPU。
- Windows 10/11 64 位：使用 PaddleOCR CPU；具备兼容环境时可自行安装 PaddlePaddle GPU 版本。
- Linux 64 位：使用 PaddleOCR CPU；NVIDIA 环境可按 PaddlePaddle 官方说明切换 GPU wheel。
- Python 3.11 或 3.12；推荐 Python 3.11。

Intel Mac 可以继续使用 macOS Vision，但 PaddlePaddle 3.3 不再提供 Intel macOS wheel。

## 硬件和磁盘

建议至少 16GB 内存。主虚拟环境、UVDoc、PaddleOCR、Pix2Text 和 UniMERNet Tiny 安装完成后通常占用 6–10GB；首次安装需要访问 GitHub、PyPI、PaddlePaddle 软件源和 Hugging Face。

## 首次安装

### macOS / Linux

安装 Git 和 Python 3.11。macOS 使用 Vision 时还需要 Xcode Command Line Tools：

```bash
xcode-select --install
```

然后执行：

```bash
git clone git@github.com:flynncat/MistakeNotebook.git
cd MistakeNotebook
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,v2]'
python scripts/setup_runtime.py
mistake-book --root .
```

### Windows PowerShell

安装 64 位 Python 3.11 和 Git，然后执行：

```powershell
git clone git@github.com:flynncat/MistakeNotebook.git
Set-Location MistakeNotebook
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,v2]"
python scripts\setup_runtime.py
mistake-book --root .
```

浏览器打开 `http://127.0.0.1:8765`。服务仅监听本机回环地址，不应直接暴露到公网。

## 按需安装运行时

统一安装命令 `python scripts/setup_runtime.py` 会：

1. 克隆并校验 UVDoc 展平模型。
2. 创建隔离的 Pix2Text 与 UniMERNet 环境，下载约 510MB 公式权重并校验 SHA-256。
3. 创建隔离的 PaddleOCR 环境并预下载中文 OCR 模型。

可以使用以下参数跳过不需要的部分：

```bash
python scripts/setup_runtime.py --skip-paddle
python scripts/setup_runtime.py --skip-formulas
python scripts/setup_runtime.py --skip-v2
python scripts/setup_runtime.py --skip-warmup
```

macOS 默认使用 Vision，因此只想保持当前 Mac 识别路径时可以跳过 PaddleOCR。Windows/Linux 必须安装 PaddleOCR，除非改用云端识别。

## OCR 策略

- `MISTAKE_BOOK_LOCAL_OCR=auto`：默认值；macOS 优先 Vision，其他平台使用 PaddleOCR。
- `MISTAKE_BOOK_LOCAL_OCR=vision`：强制 macOS Vision，仅 macOS 可用。
- `MISTAKE_BOOK_LOCAL_OCR=paddle`：三平台均强制 PaddleOCR。
- Tesseract 是可选的独立复核 OCR，不再是主 OCR 的硬依赖。安装中文语言包后系统会自动使用；找不到时题目进入人工确认，但服务仍可启动。

macOS 可用 Homebrew 安装复核 OCR：

```bash
brew install tesseract tesseract-lang
```

如果 Tesseract 位于非标准路径，设置 `TESSERACT_CMD`。

## 识别、重建与导出

- 多题页面按印刷题号切分，数字列表不会作为新题号。
- 普通文字与公式分别识别；独立变量保留为可检索文字，复杂公式使用 UniMERNet Tiny。
- 地图、圆图和方格图从原图像素恢复，并执行结构与像素支撑校验。
- 无法可靠恢复的内容不会自动通过，需要在网页中保留图像或人工修订。
- DOCX 中使用可编辑文字与 OMML 公式；Markdown 中只把无法文本化的公式和题图保存为附件。

字体优先使用微软雅黑；不可用时自动回退到系统中文字体或 Noto Sans CJK。

## 可选分类模型

本机 Ollama 可用时，系统会调用 `qwen2.5:14b-instruct` 辅助分类；不可用时回退到确定性规则，不影响 OCR：

```bash
ollama pull qwen2.5:14b-instruct
```

完全禁用 Ollama：

```bash
export MISTAKE_BOOK_DISABLE_OLLAMA=1
```

## 云端 OCR

可以切换到 OpenAI 兼容视觉接口：

```bash
export MISTAKE_BOOK_PROVIDER=cloud
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_VISION_MODEL=gpt-4.1-mini
```

云端模式会发送去除 EXIF 后的题面图。启用前请确认服务商的数据留存政策，且不要把 API key 写入仓库。

## 常用环境变量

- `MISTAKE_BOOK_DATA_DIR`：数据目录，默认是仓库下的 `data/`。
- `MISTAKE_BOOK_PIPELINE`：默认 `v2`；设置为 `v1` 可临时回退。
- `MISTAKE_BOOK_FORMULA_OCR`：默认 `1`；设置为 `0` 禁用公式模型。
- `MISTAKE_BOOK_LOCAL_OCR`：`auto`、`vision` 或 `paddle`。
- `MISTAKE_BOOK_PROVIDER`：`local` 或 `cloud`。
- `MISTAKE_BOOK_SESSION_TOKEN`：可选固定会话令牌；默认在 `data/.session-token` 自动生成。
- `PYTORCH_ENABLE_MPS_FALLBACK=1`：Apple Silicon 遇到不支持的 MPS 算子时启用 CPU 回退。

V2 默认要求人工确认后才进入“已处理资产”。`MISTAKE_BOOK_V2_ALLOW_UNVERIFIED=1` 仅用于受控测试，不建议日常开启。

## 本地数据与隐私

以下内容只保存在本机并被 Git 忽略：

- `data/`：上传原图、SQLite 数据库、分类配置、日志和导出文件。
- `.models/`：第三方模型源码、权重及隔离虚拟环境。
- `output/`：基准测试结果。
- `Sample/`、`目标效果/`：本地测试照片。

仓库只保存模型来源、固定 revision、大小和 SHA-256，不保存权重或用户照片。

## macOS 后台服务（可选）

复制 `scripts/com.cursor.mistake-book.plist.example`，将其中的 `__PROJECT_ROOT__` 替换为仓库绝对路径，再放入 `~/Library/LaunchAgents/com.cursor.mistake-book.plist`。示例文件不包含任何用户路径。

## 测试

```bash
python -m pytest
```

CI 在 macOS、Ubuntu 和 Windows 上执行不下载大型模型的安装与单元测试。本地完整模型与样本基准：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/benchmark_v2_models.py
```

报告写入 `output/v2-benchmark/`。本地 `Sample/` 不会提交，因此依赖真实照片的验收测试在 CI 中会自动跳过。

## 故障排查

- “PaddleOCR is not installed”：运行 `python scripts/setup_paddle_ocr.py`。
- “formula model is not installed”：运行 `python scripts/setup_formula_models.py`。
- macOS `xcrun` 或 `swiftc` 不存在：运行 `xcode-select --install`。
- Windows PowerShell 禁止激活脚本：可以不激活，直接使用 `.\.venv\Scripts\python.exe` 执行上述命令。
- 模型下载中断：重新运行安装脚本；公式权重支持断点续传并在完成后校验哈希。

开源方案分析和选择理由见 `docs/open-source-solution-evaluation.md`。
