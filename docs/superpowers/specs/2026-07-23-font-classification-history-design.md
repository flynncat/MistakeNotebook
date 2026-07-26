# 字体、题型分类与历史检索设计

## 目标

1. 重建题目时仅在字体判断具有可靠证据时采用匹配字体；无法可靠判断时统一使用本机黑体，禁止凭视觉猜测。
2. 每道题自动生成一级领域、二级题型、知识点摘要和分类置信度。
3. 网页支持查看全部历史题目，并按关键词、一级领域、二级题型筛选，按创建时间升序或降序排列。

## 方案

### 字体

照片 OCR 不提供字体元数据，单凭低分辨率笔画无法稳定区分具体字库。因此不把视觉猜测当成识别结果。只有输入格式未来提供明确字体元数据且能映射到本机字体文件时才视为“已识别”；当前照片输入统一进入黑体回退。

黑体文件按以下顺序选择：

1. `/System/Library/Fonts/STHeiti Light.ttc` 的简体字族索引 1；
2. `/System/Library/Fonts/STHeiti Light.ttc` 的索引 0；
3. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`；
4. `/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc`。

所有 TTC 使用索引 0。加载后渲染“错题数学123”，要求掩膜非空，且四个汉字的字形掩膜不能全部相同，以排除统一缺字方框。全部不可用或验证失败时停止该题渲染并给出“缺少中文黑体”错误，不能回退到不支持中文或字形不确定的默认字体。处理指标固定记录：

- `font.detected_family`；
- `font.rendered_family`；
- `font.confidence`；
- `font.source`；
- `font.fallback_reason`；
- `font.path`。

照片回退时 `detected_family=null`、`confidence=0`、`source=raster_no_font_metadata`、`rendered_family=黑体`。人工修订后重新渲染时复用该字体选择函数。

### 分类

`problems` 表新增并逐题持久化：

- `category_group TEXT NOT NULL DEFAULT '未分类'`；
- 复用现有可空 `category TEXT` 作为显示文本；启动迁移将 `NULL`/空值更新为“未分类”，后续写入均使用非空值；
- `category_key TEXT NOT NULL DEFAULT '未分类'` 作为规范二级题型和筛选键；
- `summary TEXT NOT NULL DEFAULT ''`；
- `category_confidence REAL NOT NULL DEFAULT 0`；
- `category_source TEXT NOT NULL DEFAULT 'automatic'`，取值为 `automatic|manual|migrated`。

一级领域固定为：计数、组合、数论、几何、应用、行程、逻辑、未分类。首批二级词表及样例映射：

- 计数：染色问题、数位进位、组数问题、排列问题、组合计数、计数综合；
- 组合：抽屉原理、容斥原理；
- 数论：整除余数、质因数、数论综合；
- 几何：面积周长、图形计数、几何综合；
- 应用：比例问题、工程问题、浓度利润、应用题；
- 行程：相遇追及、行程综合；
- 逻辑：逻辑推理。

每个规则包含带权关键词。命中分为关键词权重之和；最高分低于 3 时规则结果为未分类。置信度为 `min(0.98, 0.55 + 0.08 × (最高分-3) + 0.04 × (最高分-次高分))`。置信度不低于 0.75 时采用规则结果；低于 0.75 时调用 Ollama。Ollama 必须返回一级领域、二级题型、摘要和 0–1 置信度，且组合必须属于固定词表；仅当输出合法且模型置信度比规则高至少 0.05 时覆盖规则结果。

最终决策：

- 规则分低于 3 且 Ollama 无合法结果：保存“未分类”、置信度 0并进入待确认；
- 规则有结果但置信度低于 0.75，Ollama 未覆盖：保留规则结果并以“分类置信度较低”进入待确认；
- Ollama 输出合法但未比规则高 0.05：按上一条保留规则，不视为 Ollama 覆盖；
- 达到覆盖条件的 Ollama 结果或高置信度规则结果：正常保存。

分类在完整题干 OCR 后执行。人工修订题干但未手改分类时重新分类。人工分类请求必须同时提交一级领域和二级题型；后端校验组合属于固定词表，非法组合返回 422。合法人工分类保存为 `category=category_key=二级题型`、`category_confidence=1`、`category_source=manual`。知识点摘要保留自动分类结果；如果同时修订题干，则先根据新题干重新生成摘要，再覆盖人工一级、二级标签。

旧记录迁移不修改原 `category` 文本，只回填新增字段：

- `计数·X` → 一级“计数”，显示文本仍保留原 `category`；若 X 属于计数二级词表则 `category_key=X`，否则为“计数综合”；
- 抽屉原理 → 组合；
- 比例问题、应用题 → 应用；
- 数论问题 → 数论/数论综合；
- 几何问题 → 几何/几何综合；
- 行程问题 → 行程/行程综合；
- 其他 → 未分类；`NULL`/空值同时把显示文本更新为“未分类”。

旧记录的 `summary` 从 `metrics.recognition_summary` 回填；`category_confidence=0`、`category_source=migrated`，表示历史推断而非重新识别。

### 历史检索网页

新增 `GET /api/problems`：

- `sort=newest|oldest`，默认 `newest`；
- 可选 `category_group`、`category`；其中 `category` 匹配规范 `category_key`；
- 可选 `q`，匹配文件名和题干；
- `limit` 默认 50，范围 1–100；
- `offset` 默认 0，必须非负；
- 返回 `{items,total,limit,offset,sort}`；
- 稳定排序使用 `created_at DESC/ASC, id DESC/ASC`。

`total` 是应用分页前、应用全部筛选后的总数。每个 `items` 元素至少返回：`id,batch_id,filename,status,review_status,category_group,category,category_key,summary,category_confidence,ocr_text,confidence,created_at,updated_at,metrics,images`。

新增 `GET /api/categories`，从固定分类词表返回 `{groups:[{name,categories:[]}]}` 供筛选器使用，不因数据库历史脏值改变。历史筛选使用 `category_group/category_key`，因此旧“计数·X”和新“X”归入同一结果。查询枚举或分页参数无效时返回 422。

首页顶部增加“当前批次/全部历史”切换、搜索框、领域和题型筛选、时间排序。题目卡片显示一级领域、二级题型、创建时间和分类置信度。处理新批次时仍自动刷新当前批次；切换到历史视图后按筛选条件重新加载。历史视图不改变当前批次 PDF 导出的对象。

## 错误处理

- 没有字体元数据不阻断处理，固定回退黑体；本机没有任何受支持中文黑体时才阻断单题渲染。
- 分类器不可用时保留规则分类；无稳定分类时标记待确认。
- 历史查询参数采用白名单，排序字段不接受任意 SQL。
- 旧数据库启动时通过幂等 `ALTER TABLE` 增加缺失列并按上述映射回填，不删除记录，不覆盖原分类文本。

## 测试

- 字体无法确认、首选字体缺失、全部黑体缺失三种分支。
- 分类评分阈值 3、置信度阈值 0.75 和 Ollama 覆盖差值 0.05 的边界。
- `IMG_9323.HEIC` → 计数/组数问题；
- `IMG_9324.HEIC` → 计数/数位进位；
- `IMG_9325.HEIC`、`IMG_9327.HEIC` → 计数/染色问题；
- `IMG_9341.HEIC` → 组合/抽屉原理；
- `IMG_9345.HEIC` → 应用/比例问题。
- Ollama 不可用、超时、非法枚举和非法 JSON。
- 旧分类逐例迁移、可空分类迁移、旧新分类等价筛选，并保留非空原文本；旧记录仍可审核和导出 PDF。
- 人工只改题干时重新分类；人工改分类时置信度变为 1。
- 历史 API 覆盖稳定时间正倒序、相同时间戳的 ID 次序、领域/题型筛选、关键词、分页总数和无效参数。
- 低置信度规则保留但进入待确认，规则和 Ollama 都失败时保存未分类。
- 网页返回的新字段与当前批次轮询、审核、筛选及 PDF 导出流程兼容。
