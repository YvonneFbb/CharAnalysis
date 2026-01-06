# CharAnalysis（结构与功能备忘）

本 README 仅记录当前代码结构与功能分布，便于维护时快速定位。

## 核心结构

```
CharAnalysis/
├── src/
│   ├── review/          # 筛选/审查主模块（OCR + 切割）
│   └── analysis/        # 分析主模块（bundle + 交互可视化）
├── data/
│   ├── raw/             # 原始 PDF/图片
│   ├── results/         # 筛选与切割结果
│   ├── analysis/        # 分析包（由 collect 生成）
│   └── metadata/        # 标准字/书籍元数据
├── pipeline             # CLI 入口（转发到 src/review/pipeline.py）
└── requirements.txt
```

## 模块职责

### src/review（筛选/审查）
- `app.py`：Flask 审查服务（OCR 审查、切割审查、Fixing 页面）
- `pipeline.py`：批处理流程入口（预处理、OCR、匹配、裁切、Paddle 筛选）
- `config.py`：路径与配置
- `preprocess/`：图像预处理
- `ocr/`：OCR 识别（macOS LiveText）
- `filter/`：标准字匹配与筛选
- `crop/`：字符裁切
- `paddle/`：Paddle 筛选流程（切割 + PaddleOCR + TopK）
- `segment/`：切割与参数调整
- `utils/`：公共工具
- `web/`：前端模板与静态资源

### src/analysis（分析）
- `collect_analysis_bundle.py`：从 review 结果生成分析包（`data/analysis/`）
- `generate_book_montage.py`：每本书拼贴图（固定框）
- `time_explorer_mpl.py`：交互式分析（matplotlib，标准 boxplot）
- `_maintenance/`：维护脚本
- `_archive/`：旧脚本归档

## 数据真源与路径

- 审查结果真源：`data/results/manual/review_books/*.json`
- Paddle 结果：`data/results/paddle/review_books/*.json`
- 切割图片：`data/results/manual/segmented/`
- Paddle 切割图片：`data/results/paddle/segmented/`
- 分析包输入：`data/analysis/`（由 collect 生成）
- 书籍元数据：`data/metadata/books_metadata.csv`
- 标准字：`data/metadata/standard_chars.json`

## 入口

- 审查服务：`python src/review/app.py`
- 批处理入口：`./pipeline`
- 交互分析：`python src/analysis/time_explorer_mpl.py --bundle data/analysis`
