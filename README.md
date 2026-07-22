# 小奥错题集

本地处理错题照片：校正方向、透视和纸面弯曲，保守清除不覆盖印刷内容的笔迹，识别并动态归类题目，最后生成 A4 错题集 PDF。

## 启动

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/mistake-book --root .
```

浏览器打开 `http://127.0.0.1:8765`。首页可以上传照片，也可以直接处理 `Sample` 目录。

## 识别模式

默认在 macOS 上使用系统 Vision OCR，并在本机 Ollama 可用时调用 `qwen2.5:14b-instruct` 归类；Ollama 不可用时回退到确定性小奥关键词分类。

云端 OpenAI 兼容接口：

```bash
export MISTAKE_BOOK_PROVIDER=cloud
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_VISION_MODEL=gpt-4.1-mini
```

云端模式只发送去除 EXIF 后、已校正但尚未去笔迹的题面图。切换云端前请确认所用服务商的数据留存政策。

## 数据

原图、中间图片、SQLite 数据库和 PDF 默认保存在本地 `data/`，不会提交到 Git。第一版只支持一张照片对应一道题；无法确认安全的笔迹会保留，并进入集中确认队列。

## 测试

```bash
.venv/bin/pytest
```

样例回归：

```bash
.venv/bin/python scripts/evaluate_samples.py
```

报告和处理后的图片写入 `output/sample-evaluation/`。
