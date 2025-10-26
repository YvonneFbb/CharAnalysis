# 古籍字形分析系统

基于《大字结构八十四法》的古籍字形识别与分析工具

## 📋 项目概述

本项目旨在对历代古籍刻本图片进行文字识别与字形分析，通过标准字集驱动的方式，实现不同书籍之间的字形对比分析。

### 核心特性

- **🔍 整体 OCR**：基于 macOS LiveText 的高精度文字识别
- **📚 标准字驱动**：333 个标准汉字（来自《大字结构八十四法》）
- **⚡ 多线程处理**：支持并发处理，大幅提升处理速度（2-4倍）
- **🎯 智能筛选**：从 OCR 结果中提取标准字，按书籍组织
- **🌐 Web 审查界面**：Flask + 按需裁切，无需预生成图片
- **💾 进度追踪**：自动保存进度，支持断点续传

### 处理流程

```
原始 PDF 古籍 → PDF转换 → 预处理 → OCR识别 → 标准字匹配 → Web审查 → 导出结果
```

## 🚀 快速开始

### 环境要求

- **操作系统**：macOS（OCR 依赖 macOS LiveText）
- **Python**：3.10+
- **依赖**：OpenCV, PyMuPDF, Flask, ocrmac, etc.

### 安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd CharAnalysis

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
./pipeline config
```

### 完整工作流程

```bash
# 激活虚拟环境
source .venv/bin/activate

# 步骤 1: PDF 转图片（多线程）
./pipeline convert data/raw/ --max-volumes 5 --workers 4

# 步骤 2: 图像预处理（多线程）
./pipeline preprocess data/raw/ --max-volumes 5 --workers 6

# 步骤 3: OCR 识别（多线程）
./pipeline ocr data/results/preprocessed/ --max-volumes 5 --workers 8

# 或者使用 all 命令一次完成预处理+OCR
./pipeline all data/raw/ --max-volumes 5 --workers 6

# 步骤 4: 匹配标准字
./pipeline match data/results/ocr/

# 步骤 5: 启动审查界面
cd src/review
python3 app.py
# 访问 http://localhost:5001/review

# 步骤 6: 在 Web 界面中审查并导出结果
```

## 📖 详细文档

### 命令行工具

查看所有可用命令：

```bash
./pipeline --help
```

主要命令：

| 命令 | 说明 | 示例 |
|------|------|------|
| `convert` | PDF 转图片 | `./pipeline convert data/raw/ --workers 4` |
| `preprocess` | 图像预处理 | `./pipeline preprocess data/raw/ --workers 6` |
| `ocr` | OCR 识别 | `./pipeline ocr data/results/preprocessed/ --workers 8` |
| `all` | 预处理+OCR | `./pipeline all data/raw/ --workers 6` |
| `match` | 匹配标准字 | `./pipeline match data/results/ocr/` |
| `crop` | 裁切字符 | `./pipeline crop data/results/matched_by_book.json --book "书名"` |
| `config` | 显示配置 | `./pipeline config` |

### 多线程配置

所有批处理命令都支持 `--workers` 参数：

```bash
# PDF 转换：4-6 线程（I/O 密集）
./pipeline convert data/raw/ --workers 4

# 预处理：6-7 线程（CPU 密集，核心数 - 1）
./pipeline preprocess data/raw/ --workers 6

# OCR：8-12 线程（I/O+CPU 混合，核心数 × 1.5）
./pipeline ocr data/results/preprocessed/ --workers 10
```

**推荐配置**（8核 CPU）：
- PDF 转换：4线程
- 预处理：6-7线程
- OCR：8-10线程

### 进度跟踪

所有批处理命令都支持进度跟踪和断点续传：

```bash
# 正常处理（自动保存进度）
./pipeline preprocess data/raw/ --workers 6

# 中断后重新运行，会从上次停止的地方继续
./pipeline preprocess data/raw/ --workers 6

# 强制重新处理所有文件（忽略进度）
./pipeline preprocess data/raw/ --workers 6 --force
```

### Web 审查界面

启动审查服务器：

```bash
cd src/review
python3 app.py
```

访问：http://localhost:5001/review

**核心功能**：
- 选择书籍（38本古籍）
- 按 84 法分类浏览字符
- 单击选择/取消选择实例
- 分页浏览（每页 20 个实例）
- 实时进度显示
- 导出/导入审查结果

详细使用说明：[审查界面使用说明.md](审查界面使用说明.md)

## 📁 项目结构

```
CharAnalysis/
├── src/                        # 源代码
│   ├── pipeline.py             # 主程序（CLI 接口）
│   ├── config.py               # 配置管理
│   ├── preprocess/             # 图像预处理
│   ├── ocr/                    # OCR 识别
│   ├── filter/                 # 标准字匹配
│   ├── crop/                   # 字符裁切
│   ├── review/                 # Web 审查界面
│   └── utils/                  # 工具模块
├── data/                       # 数据目录
│   ├── standard_chars.json     # 标准字集（84法分类）
│   ├── raw/                    # 原始 PDF/图片
│   └── results/                # 处理结果
│       ├── preprocessed/       # 预处理图片
│       ├── ocr/                # OCR 结果 JSON
│       ├── matched_by_book.json # 匹配结果
│       └── review_app.html     # 审查界面
├── pipeline                    # 命令行工具
├── requirements.txt            # Python 依赖
├── README.md                   # 本文件
├── CLAUDE.md                   # 开发备忘
└── CHANGELOG_20251025.md       # 更新日志
```

## 🛠️ 技术栈

- **Python 3.10+**：主要开发语言
- **OpenCV**：图像处理（CLAHE、形态学、滤波）
- **PyMuPDF (fitz)**：PDF 转图片
- **ocrmac**：macOS LiveText OCR 封装
- **Flask**：Web 审查界面后端
- **NumPy**：数值计算
- **tqdm**：进度条显示
- **concurrent.futures**：多线程处理

## 📊 数据统计

基于当前处理的数据：

- **书籍数量**：38 本古籍
- **标准字集**：333 个字符（84 个分类法）
- **字符覆盖率**：324/333（97.3%）
- **总实例数**：约 1,000,000+ 个字符实例

## 🎯 核心设计

### 1. 标准字驱动

- 从《大字结构八十四法》提取 333 个标准字
- 84 个书法结构分类
- 数据量可控，支持人工审查

### 2. 按书籍组织

- 每本书独立处理和统计
- 便于增量处理
- 便于字形对比分析

### 3. 按需裁切

- 无需预生成 100万+ 张图片
- 实时裁切，节省存储空间
- 加载速度快（每页仅 20 个实例）

### 4. 多线程处理

- 2-4倍速度提升
- 线程安全的进度跟踪
- 完全向后兼容

## ⚠️ 注意事项

1. **macOS 依赖**：ocrmac 只能在 macOS 上运行
2. **线程数配置**：不是越多越好，建议不超过 CPU 核心数 × 1.5
3. **内存占用**：大量并发处理时注意内存占用
4. **进度文件**：`.progress` 文件记录处理进度，可手动删除以重新开始

## 📝 示例

### 处理单本书

```bash
# 1. PDF 转图片
./pipeline convert data/raw/01_1127_尚书正义/ --workers 2

# 2. 预处理
./pipeline preprocess data/raw/01_1127_尚书正义/ --workers 4

# 3. OCR
./pipeline ocr data/results/preprocessed/01_1127_尚书正义/ --workers 6

# 4. 匹配（会匹配所有书籍）
./pipeline match data/results/ocr/

# 5. 裁切（只裁切这本书）
./pipeline crop data/results/matched_by_book.json --book "01_1127_尚书正义"
```

### 批量处理多本书

```bash
# 限制每本书只处理前 5 册，使用 4 线程
./pipeline convert data/raw/ --max-volumes 5 --workers 4
./pipeline all data/raw/ --max-volumes 5 --workers 6
./pipeline match data/results/ocr/
```

## 🔗 相关文档

- [CLAUDE.md](CLAUDE.md) - 项目开发备忘
- [审查界面使用说明.md](审查界面使用说明.md) - Web 审查界面使用指南
- [CHANGELOG_20251025.md](CHANGELOG_20251025.md) - 最新更新日志

## 📄 许可证

[许可证信息待补充]

## 👥 贡献

欢迎提交 Issues 和 Pull Requests！

---

**最后更新**：2025-10-25
**版本**：2.0.0（支持多线程处理）
