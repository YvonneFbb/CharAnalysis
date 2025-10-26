# CLAUDE.md - 项目开发备忘

> 本文件用于记录项目的核心信息、开发进展和设计决策，方便 Claude 在后续对话中快速了解项目状态。

## 项目概述

### 目标

对历代古籍刻本图片进行文字识别与字形分析，基于《大字结构八十四法》的标准字集，实现不同书籍之间的字形对比。

### 核心思路

**旧方案的问题**：
- 先分割所有字符再分析
- 不同书之间的字形差异大，缺少统一的对比标准
- 数据量巨大，难以人工检查和优化

**新方案的改进**：
- **整体 OCR**：直接对整张图进行 OCR，无需字符分割
- **标准字驱动**：预先定义 333 个标准汉字（来自《大字结构八十四法》）
- **后筛选**：从 OCR 结果中只提取标准字集中的字符
- **统一标准**：不同书籍使用相同的标准字进行对比分析
- **数据量可控**：只处理标准字，支持人工检查优化
- **多线程处理**：支持并发处理，大幅提升处理速度

### 完整处理流程

```
原始 PDF 古籍
    ↓
[PDF 转换] PyMuPDF (支持多线程)
    ↓
原始古籍图片
    ↓
[预处理] CLAHE + 墨色增强 + 去噪 (支持多线程)
    ↓
预处理图片
    ↓
[整体 OCR] macOS LiveText (支持多线程)
    ↓
OCR 结果（JSON：文字 + 位置 + 置信度）
    ↓
[标准字匹配] 只保留标准字集中的字符（按书籍组织）
    ↓
[Web 审查界面] Flask + 按需裁切
    ↓
人工审查确认
    ↓
[导出结果] JSON 格式的审查结果
```

## 项目结构

```
CharAnalysis/
├── src/
│   ├── config.py              # 配置管理（路径、预处理、OCR 参数）
│   ├── pipeline.py            # 主流程（CLI 接口，支持多线程）
│   ├── preprocess/
│   │   └── core.py            # 图像预处理（CLAHE、墨色保持、多线程）
│   ├── ocr/
│   │   └── livetext.py        # OCR 识别（macOS LiveText、多线程）
│   ├── filter/
│   │   └── match_standard_chars.py  # 标准字匹配（按书籍组织）
│   ├── crop/
│   │   └── crop_characters.py       # 字符裁切（支持按书籍裁切）
│   ├── review/
│   │   ├── app.py             # Flask 审查服务器
│   │   └── __init__.py
│   └── utils/
│       ├── path.py            # 路径工具
│       ├── progress.py        # 进度跟踪
│       ├── file_filter.py     # 文件过滤
│       ├── pdf_converter.py   # PDF转换（支持多线程）
│       └── parallel.py        # 并发处理工具
├── data/
│   ├── standard_chars.json    # 标准字集（结构化：84法分类）
│   ├── standard_chars.txt     # 标准字集（纯文本：333字）
│   ├── raw/                   # 原始 PDF/图片
│   └── results/
│       ├── preprocessed/      # 预处理结果
│       ├── ocr/               # OCR 结果（JSON）
│       ├── matched_by_book.json  # 按书籍组织的匹配结果
│       └── review_app.html    # 审查界面（静态版）
├── pipeline                   # 可执行脚本
├── requirements.txt           # Python 依赖
├── README.md                  # 项目文档
├── CLAUDE.md                  # 本文件
├── 审查界面使用说明.md         # 审查系统使用指南
└── CHANGELOG_20251025.md      # 更新日志
```

## 开发进展

### ✅ 已完成（Phase 1 - 基础设施）

1. **标准字集准备**
   - 从《大字结构八十四法》提取 84 个分类、333 个不重复标准汉字
   - 保存为 JSON（带分类）和 TXT（纯文本）格式

2. **项目基础设施**
   - 虚拟环境 `.venv` + 依赖管理
   - 目录结构搭建
   - `.gitignore` 配置

3. **配置管理模块** (`src/config.py`)
   - 路径配置（数据目录、结果目录）
   - 预处理参数（CLAHE、墨色保持、去噪、断笔修补）
   - OCR 参数（framework、识别级别、语言偏好）
   - 配置校验和摘要功能

4. **预处理模块** (`src/preprocess/core.py`)
   - CLAHE 对比度自适应增强
   - 黑帽墨色保持 + 反锐化掩膜
   - 可选双边滤波去纸纹
   - 可选断笔修补（闭运算）
   - 支持单文件和目录批处理
   - **多线程并发处理**（`--workers` 参数）

5. **OCR 模块** (`src/ocr/livetext.py`)
   - 基于 macOS LiveText
   - 输出完整 JSON：文字、置信度、bbox（像素和归一化坐标）
   - 支持单文件和目录批处理
   - **多线程并发处理**（`--workers` 参数）

6. **PDF 转换模块** (`src/utils/pdf_converter.py`)
   - 基于 PyMuPDF (fitz)
   - 支持单个 PDF 和批量转换
   - 支持 max-volumes 限制
   - 进度跟踪和断点续传
   - **多线程并发处理**（`--workers` 参数）

7. **Pipeline 主程序** (`pipeline` + `src/pipeline.py`)
   - CLI 接口：`convert`、`pdf2images`、`preprocess`、`ocr`、`all`、`match`、`crop`、`config`
   - 所有批处理命令支持 `--workers` 参数
   - 单文件和批量处理
   - 完整的帮助文档

### ✅ 已完成（Phase 2 - 标准字筛选与审查）

1. **标准字匹配模块** (`src/filter/match_standard_chars.py`)
   - 从 OCR 结果 JSON 中筛选出标准字集中的字符
   - **按书籍组织**结果（每本书独立统计）
   - 生成 `matched_by_book.json`（包含字符位置、册号、页码信息）
   - 提供字符覆盖率统计（324/333 标准字在 38 本书中被找到）

2. **字符裁切模块** (`src/crop/crop_characters.py`)
   - 根据 bbox 坐标从预处理图中裁切字符
   - 支持 `--book` 参数指定单本书裁切
   - 保存格式：`data/results/chars/{book_name}/{char}/册XX_pageXXXX_index.png`
   - 支持可配置的边界填充（默认 5 像素）

3. **Web 审查界面** (`src/review/app.py` + `data/results/review_app.html`)
   - **Flask 后端服务器**（端口 5001）
   - **按需裁切**：只在查看时实时裁切图片，无需预生成
   - **84法分类排序**：字符按《大字结构八十四法》顺序展示
   - **分页加载**：每页 20 个实例，优化性能
   - **进度追踪**：显示已审查字符数和完成百分比
   - **导出/导入**：支持审查结果的 JSON 导出和导入
   - **localStorage 持久化**：浏览器自动保存进度
   - **简约线条设计**：现代化的极简 UI

4. **Pipeline 命令扩展**
   - `./pipeline match <ocr_dir>` - 匹配标准字
   - `./pipeline crop <matched_json>` - 裁切字符（支持 --book）
   - 审查界面通过独立的 Flask 服务器启动

### 🎯 当前状态（Phase 3 - 生产就绪）

**完整工作流已打通**：
```bash
# 1. PDF 转图片（多线程）
./pipeline convert data/raw/ --max-volumes 5 --workers 4

# 2. 预处理（多线程）
./pipeline preprocess data/raw/ --max-volumes 5 --workers 6

# 3. OCR 识别（多线程）
./pipeline ocr data/results/preprocessed/ --max-volumes 5 --workers 8

# 4. 匹配标准字
./pipeline match data/results/ocr/

# 5. 启动审查界面
cd src/review && python3 app.py
# 访问 http://localhost:5001/review

# 6. 导出审查结果
# 在 Web 界面中点击"导出审查结果"
```

**数据统计**（基于当前数据）：
- 书籍数量：38 本古籍
- 标准字集：333 个字符（84 个分类法）
- 字符覆盖率：324/333（97.3%）
- 总实例数：约 1,000,000+ 个字符实例

## 关键设计决策

### 1. 为什么使用整体 OCR 而非字符分割？

**优势**：
- 简化流程，无需复杂的分割算法
- LiveText 识别准确度高
- 获得完整的文本上下文信息
- 可以利用上下文提高识别准确率

**劣势**：
- 依赖 macOS 平台
- 对于重叠、模糊的字符可能识别不准

**决策**：优势大于劣势，当前方案更适合快速迭代

### 2. 为什么只处理标准字集？

**原因**：
- 旧方案数据量太大（每本书数千字），无法人工检查
- 不同书之间缺少统一对比标准
- 标准字集（333字）数据量适中，支持精细化分析

**标准字来源**：《大字结构八十四法》
- 84 个书法结构分类
- 每类 4 个左右例字
- 涵盖常见书法结构特征

### 3. 为什么采用按书籍组织的匹配结果？

**优势**：
- 便于按书籍进行人工审查
- 每本书的字形特征独立
- 支持增量处理（单本书重新处理）
- 便于统计分析（每本书的标准字覆盖率）

**数据结构**：
```json
{
  "books": {
    "书名": {
      "chars": {
        "字": [{"volume": 1, "page": "page_0001", "bbox": {...}}]
      }
    }
  }
}
```

### 4. 为什么使用按需裁切而非预生成图片？

**按需裁切的优势**：
- **节省存储空间**：不需要预先生成 100 万+ 张图片
- **加载速度快**：只裁切当前查看的 20 个实例
- **灵活性高**：可以随时调整裁切参数（padding）
- **开发效率**：无需等待漫长的预裁切过程

**实现方式**：
- Flask 后端提供 `/api/crop` 接口
- 前端异步请求裁切图片
- 返回 base64 编码的图片数据

### 5. 为什么添加多线程支持？

**性能需求**：
- 38 本书，每本书 2-10 册，每册 100-300 页
- 总计约 10,000+ 张图片需要处理
- 单线程处理时间过长（数天）

**多线程收益**：
- **PDF 转换**：I/O 密集，4线程 ≈ 2-3倍速度
- **预处理**：CPU 密集，4线程 ≈ 3-4倍速度
- **OCR**：I/O+CPU 混合，4线程 ≈ 2-3倍速度

**实现方式**：
- 使用 `ThreadPoolExecutor` 实现
- 线程安全的进度跟踪（`threading.Lock`）
- 向后兼容（默认 `workers=1`）

## 技术栈

- **Python 3.10+**：主要开发语言
- **OpenCV**：图像处理（CLAHE、形态学、滤波）
- **PyMuPDF (fitz)**：PDF 转图片
- **ocrmac**：macOS LiveText OCR 封装
- **Flask**：Web 审查界面后端
- **NumPy**：数值计算
- **Pillow**：图像 I/O
- **tqdm**：进度条显示
- **concurrent.futures**：多线程处理

## 使用方法

### 完整工作流程

```bash
# 0. 激活虚拟环境
source .venv/bin/activate

# 1. PDF 转图片（支持多线程）
./pipeline convert data/raw/ --max-volumes 5 --workers 4

# 2. 预处理（支持多线程）
./pipeline preprocess data/raw/ --max-volumes 5 --workers 6

# 3. OCR 识别（支持多线程）
./pipeline ocr data/results/preprocessed/ --max-volumes 5 --workers 8

# 或者使用 all 命令一次完成预处理+OCR
./pipeline all data/raw/ --max-volumes 5 --workers 6

# 4. 匹配标准字
./pipeline match data/results/ocr/ -o data/results/matched_by_book.json

# 5. （可选）裁切特定书籍的字符
./pipeline crop data/results/matched_by_book.json --book "01_1127_尚书正义"

# 6. 启动审查界面
cd src/review
python3 app.py
# 访问 http://localhost:5001/review

# 7. 在 Web 界面中审查并导出结果
```

### 常用命令

```bash
# 查看配置
./pipeline config

# 查看帮助
./pipeline --help
./pipeline convert --help
./pipeline preprocess --help
./pipeline ocr --help
./pipeline all --help
./pipeline match --help
./pipeline crop --help

# 单文件处理
./pipeline preprocess data/raw/demo.jpg
./pipeline ocr data/results/preprocessed/demo_preprocessed.jpg

# 批量处理（推荐使用多线程）
./pipeline preprocess data/raw/ --workers 6
./pipeline ocr data/results/preprocessed/ --workers 8
```

## 审查界面使用

详细使用说明请参考：[审查界面使用说明.md](审查界面使用说明.md)

**核心功能**：
1. **选择书籍**：从 38 本书中选择
2. **浏览字符**：按 84 法分类展示，显示"已审查"标记
3. **审查实例**：单击选择/取消选择，分页浏览
4. **保存进度**：自动保存到 localStorage
5. **导出结果**：导出 JSON 格式的审查结果
6. **导入恢复**：导入之前的审查结果

## OCR 结果格式

```json
{
  "success": true,
  "image_path": "...",
  "image_size": {"width": 1920, "height": 1080},
  "timestamp": "2025-10-25T...",
  "character_count": 156,
  "full_text": "识别出的完整文本",
  "characters": [
    {
      "index": 0,
      "text": "字",
      "confidence": 0.95,
      "bbox": {"x": 100, "y": 200, "width": 50, "height": 60},
      "normalized_bbox": {"x": 0.052, "y": 0.185, "width": 0.026, "height": 0.056}
    }
  ]
}
```

## 匹配结果格式

```json
{
  "books": {
    "01_1127_尚书正义": {
      "book_name": "01_1127_尚书正义",
      "total_standard_chars": 293,
      "total_instances": 34942,
      "chars": {
        "宣": [
          {
            "volume": 2,
            "page": "page_0017",
            "char_index": 156,
            "bbox": {"x": 100, "y": 200, "width": 50, "height": 60},
            "normalized_bbox": {...},
            "confidence": 0.95,
            "source_image": "data/results/preprocessed/01_1127_尚书正义/册02_pages/page_0017_preprocessed.png"
          }
        ]
      }
    }
  },
  "summary": {
    "total_books": 38,
    "total_standard_chars_found": 324,
    "chars_coverage": {"宣": 25, "大": 38, ...}
  }
}
```

## 性能优化

### 多线程推荐配置

**CPU 核心数判断**：
```bash
# macOS
sysctl -n hw.ncpu

# 假设 8 核 CPU
```

**推荐线程数**：
- **PDF 转换**：4-6 线程（I/O 密集）
- **预处理**：6-7 线程（CPU 密集，核心数 - 1）
- **OCR**：8-12 线程（I/O+CPU 混合，核心数 × 1.5）

**示例**（8核 CPU）：
```bash
./pipeline convert data/raw/ --workers 4
./pipeline preprocess data/raw/ --workers 6
./pipeline ocr data/results/preprocessed/ --workers 10
```

### 进度跟踪与断点续传

所有批处理命令都支持进度跟踪：
- 自动保存进度到 `.progress` 文件
- 中断后重新运行会从上次停止的地方继续
- 使用 `--force` 参数强制重新处理所有文件

## 注意事项

1. **macOS 依赖**：ocrmac 只能在 macOS 上运行
2. **图片质量**：预处理效果依赖原图质量，可能需要调整参数
3. **标准字集**：当前 333 个字，如需扩展需修改 `data/standard_chars.json`
4. **OCR 准确度**：LiveText 识别准确度较高，但对模糊、重叠字符可能失败
5. **多线程限制**：线程数不是越多越好，需根据 CPU 核心数合理配置
6. **内存占用**：大量并发处理时注意内存占用，建议不超过物理核心数 × 1.5

## 参考资料

- 旧方案代码：`ref/charsis/`
- 标准字来源：`大字结构八十四法.docx`
- OpenCV 文档：https://docs.opencv.org/
- ocrmac 库：https://github.com/straussmaximilian/ocrmac
- PyMuPDF 文档：https://pymupdf.readthedocs.io/
- Flask 文档：https://flask.palletsprojects.com/

## 更新日志

- **2025-10-25**：
  - 添加多线程支持（convert、preprocess、ocr）
  - 优化审查界面（简约线条设计）
  - 修复 HTML 文件残留代码
  - 完善文档和使用指南

- **2025-10-24**：
  - 完成标准字匹配模块
  - 完成字符裁切模块
  - 完成 Flask Web 审查界面
  - 实现按需裁切功能

---

**最后更新**：2025-10-25
**当前阶段**：Phase 2 完成，Phase 3 生产就绪
**下一步**：大规模数据处理和字形分析
