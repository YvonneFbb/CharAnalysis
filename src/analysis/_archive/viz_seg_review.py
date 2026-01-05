#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于第二轮审查特征（src/analysis/_archive/analyze_seg_review.py 产物）进行可视化。

参考配置：ref/charsis/data/results/analysis/analysis_config.yaml

先实现“时间维度”：
  - 对配置中的每个分组（如含多个子分组），为每个子分组单独绘制一张箱形图（按书对比）
  - 数据源：data/analysis/features/*.json（默认）
  - 特征：默认 aspect_ratio（可通过 --feature 指定）
  - 粒度：默认“每字聚合（median）”，可通过 --per-char/--agg 调整

用法：
  python src/analysis/_archive/viz_seg_review.py \
    --config data/analysis/analysis_config.yaml \
    --features-dir data/analysis/features \
    --output data/analysis/visualizations \
    --feature aspect_ratio --per-char --agg median
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'配置文件不存在: {path}')
    if yaml is None:
        raise RuntimeError('缺少 PyYAML 依赖，请安装 pyyaml 或提供 JSON 配置')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def safe_label(name: str) -> str:
    return str(name).replace('/', '_').replace(' ', '_')


def scan_features(features_dir: Path) -> Dict[str, Path]:
    """扫描 features 目录，返回 book_name -> json_path 的映射。"""
    mapping: Dict[str, Path] = {}
    if not features_dir.exists():
        return mapping
    for p in sorted(features_dir.glob('*.json')):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            book = data.get('book_name') or p.stem
            mapping[str(book)] = p
        except Exception:
            continue
    return mapping


def infer_book_id(book_name: str) -> Optional[str]:
    """从书名前缀中提取两位编号（如 01_... → 01）。失败返回 None。"""
    m = re.match(r'^(\d{2})[_\-].*', book_name)
    return m.group(1) if m else None


def make_display_label(book_name: str, book_names_map: Optional[Dict[str, str]]) -> str:
    bid = infer_book_id(book_name)
    if bid and isinstance(book_names_map, dict) and bid in book_names_map:
        return f"{bid}_{book_names_map[bid]}"
    return book_name

def resolve_books_for_subgroup(books_spec: List[str], bookname_to_path: Dict[str, Path]) -> List[str]:
    """根据配置的 books（通常是编号列表，如 ["01","02"...]），解析为真实书名列表。"""
    # 构建 id->full book name 的映射
    id_to_book: Dict[str, str] = {}
    for bn in bookname_to_path.keys():
        bid = infer_book_id(bn)
        if bid:
            id_to_book[bid] = bn

    resolved: List[str] = []
    # 特殊：包含 ALL 或 * 时，返回全部书籍
    if any(spec in ("ALL", "*") for spec in books_spec):
        return list(bookname_to_path.keys())
    for spec in books_spec:
        if spec in id_to_book:
            resolved.append(id_to_book[spec])
        else:
            # 尝试前缀匹配（万一 id 有 3 位等）或直接把 spec 当作书名
            found = [bn for bn in bookname_to_path.keys() if bn.startswith(spec)]
            if found:
                resolved.extend(found)
            else:
                # 直接使用 spec（当它就是完整书名时）
                if spec in bookname_to_path:
                    resolved.append(spec)
    # 去重但保持顺序
    seen = set()
    out = []
    for x in resolved:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_values_for_book(features_json: Dict[str, Any], feature_name: str, per_char: bool, agg: str) -> List[float]:
    chars = features_json.get('characters', {})
    values: List[float] = []
    if not per_char:
        for ch, chdata in chars.items():
            for sample in chdata.get('samples', []):
                v = sample.get('features', {}).get(feature_name)
                if v is None:
                    continue
                try:
                    values.append(float(v))
                except Exception:
                    pass
        return values

    # 每字聚合
    import numpy as np
    for ch, chdata in chars.items():
        vals = []
        for sample in chdata.get('samples', []):
            v = sample.get('features', {}).get(feature_name)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        if not vals:
            continue
        if agg == 'mean':
            values.append(float(np.mean(vals)))
        else:
            values.append(float(np.median(vals)))
    return values


# ==================== 趋势评估（非线性/单调） ====================
def _rankdata(a: List[float]) -> np.ndarray:
    """平均秩（处理 ties），返回与 a 等长的秩向量（float）。"""
    arr = np.asarray(a, dtype=float)
    n = arr.size
    sorter = np.argsort(arr, kind='mergesort')
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        ai = arr[sorter[i]]
        while j + 1 < n and arr[sorter[j + 1]] == ai:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[sorter[i:j + 1]] = rank
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r


def kendall_tau_b(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 2:
        return None
    C = D = 0
    ties_x = ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = np.sign(x[j] - x[i])
            dy = np.sign(y[j] - y[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0 and dy != 0:
                ties_x += 1
                continue
            if dy == 0 and dx != 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                C += 1
            elif dx * dy < 0:
                D += 1
    denom = np.sqrt((C + D + ties_x) * (C + D + ties_y))
    if denom == 0:
        return None
    return float((C - D) / denom)


def theil_sen_slope(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 2:
        return None
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue
            slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return None
    return float(np.median(np.array(slopes)))


def _bootstrap_ci(stat_fn, x: List[float], y: List[float], B: int = 400, alpha: float = 0.05, seed: int = 12345) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Bootstrap CI for a statistic on (x,y). 返回 (est, low, high)。"""
    n = len(x)
    if n != len(y) or n < 3:
        return None, None, None
    rng = np.random.default_rng(seed)
    est = stat_fn(x, y)
    if est is None:
        return None, None, None
    samples = []
    idx_base = np.arange(n)
    for _ in range(B):
        idx = rng.choice(idx_base, size=n, replace=True)
        xv = [x[i] for i in idx]
        yv = [y[i] for i in idx]
        val = stat_fn(xv, yv)
        if val is not None and np.isfinite(val):
            samples.append(val)
    if not samples:
        return float(est), None, None
    s = np.array(samples, dtype=float)
    lo = float(np.percentile(s, 100 * (alpha / 2)))
    hi = float(np.percentile(s, 100 * (1 - alpha / 2)))
    return float(est), lo, hi


def plot_scatter_for_group(group_name: str,
                           book_labels: List[str],
                           books_values: List[List[float]],
                           out_path: Path,
                           title: str,
                           ylabel: str,
                           colors_for_points: Optional[List[str]] = None,
                           legend: Optional[List[Tuple[str, str]]] = None,
                           style: Optional[Dict[str, Any]] = None,
                           cluster_boundaries: Optional[List[int]] = None,
                           note_text: Optional[str] = None) -> None:
    # 样式
    figsize = (12, 6)
    dpi = 300
    if style:
        sz = style.get('figsize')
        if isinstance(sz, list) and len(sz) == 2:
            figsize = (float(sz[0]), float(sz[1]))
        dpi = int(style.get('dpi') or dpi)
        font = style.get('font_family')
        if font:
            try:
                matplotlib.rcParams['font.sans-serif'] = [font, 'Arial Unicode MS', 'SimHei', 'PingFang SC']
                matplotlib.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass

    # 动态宽度：每本书 ~1.5 英寸
    fig_width = max(figsize[0], max(12, int(len(book_labels) * 1.5)))
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]), dpi=dpi)

    # 绘制散点（所有样本或每字聚合的值）+ 均值±标准差（误差棒）
    x_positions = np.arange(len(book_labels), dtype=float)
    rng = np.random.default_rng(42)
    jitter_width = 0.15  # 更窄的抖动宽度，使散点更密集

    # 记录每本书的均值/标准差，用于相关性计算
    mu_list: List[float] = []
    sd_list: List[float] = []
    x_numeric: List[float] = []

    for i, vals in enumerate(books_values):
        if not vals:
            continue
        color = None
        if colors_for_points and i < len(colors_for_points):
            color = colors_for_points[i]
        # 根据样本量动态调整点大小和透明度，避免遮挡
        n = len(vals)
        point_alpha = 0.35 if n < 150 else (0.25 if n < 400 else 0.15)
        point_size = 10 if n < 150 else (8 if n < 400 else 6)
        # 抖动后的 x
        xs = x_positions[i] + (rng.uniform(-0.5, 0.5, size=n) * jitter_width)
        ax.scatter(
            xs, vals,
            s=point_size,
            alpha=point_alpha,
            color=color or '#4A90E2',
            edgecolors='none',
            zorder=1,
            rasterized=True,
        )
        # 均值与标准差
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        mu_list.append(mu)
        sd_list.append(sd)
        # x 数值：优先从标签中解析书籍编号，否则用序号
        try:
            import re as _re
            m = _re.match(r"^(\d+)", str(book_labels[i]))
            x_numeric.append(float(m.group(1)) if m else float(i))
        except Exception:
            x_numeric.append(float(i))
        # 先画白色底的误差棒，增强对比度
        ax.errorbar(
            x_positions[i], mu, yerr=sd,
            fmt='none',
            ecolor='white',
            elinewidth=3.2,
            capsize=5,
            alpha=0.95,
            zorder=3,
        )
        # 再覆盖彩色均值点 + 误差棒
        ax.errorbar(
            x_positions[i], mu, yerr=sd,
            fmt='D',
            color=color or 'black',
            ecolor=color or 'black',
            elinewidth=1.6,
            capsize=4,
            markersize=7,
            alpha=0.98,
            zorder=4,
        )
        # 数值标注（μ 与 σ）
        try:
            ax.annotate(
                f"μ={mu:.2f}\nσ={sd:.2f}",
                xy=(x_positions[i], mu),
                xytext=(6, 8), textcoords='offset points',
                ha='left', va='bottom', fontsize=8,
                color='black', zorder=5,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2')
            )
        except Exception:
            pass

    # 轴与标签
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(book_labels, rotation=30, ha='right')
    ax.tick_params(axis='both', which='both', labelsize=9)
    # 聚类分隔线（如有需要）
    if cluster_boundaries:
        for b in cluster_boundaries:
            xpos = (b - 0.5)
            if 0 < xpos < len(book_labels):
                ax.axvline(x=xpos, color='gray', linestyle='--', alpha=0.25, linewidth=1)
    # 网格线增强可读性
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 图例
    if legend:
        handles = [mpatches.Patch(color=color, label=name) for name, color in legend]
        # 均值±标准差 图例句柄
        mean_handle = mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=7, label='均值±标准差')
        handles.append(mean_handle)
        ax.legend(handles=handles, loc='best', fontsize=9)

    # 趋势（非线性）：仅 Spearman（均值与方差），在非分簇场景下注释到图中
    try:
        if not cluster_boundaries:
            parts = []
            # Spearman ρ(μ,x)
            est, lo, hi = _bootstrap_ci(spearman_rho, x_numeric, mu_list, B=400)
            if est is not None:
                parts.append(f"Spearman ρ(μ,x)={est:.3f}{'' if lo is None else f' 95%CI[{lo:.3f},{hi:.3f}]'} n={len(x_numeric)}")
            # Spearman ρ(σ,x)
            if len(sd_list) == len(x_numeric):
                est2, lo2, hi2 = _bootstrap_ci(spearman_rho, x_numeric, sd_list, B=400)
                if est2 is not None:
                    parts.append(f"Spearman ρ(σ,x)={est2:.3f}{'' if lo2 is None else f' 95%CI[{lo2:.3f},{hi2:.3f}]'} n={len(x_numeric)}")
            if parts:
                txt = "\n".join(parts)
                ax.text(
                    0.98, 0.02, txt,
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')
                )
                print(f"  - {group_name} Spearman summary:\n    " + txt.replace("\n", "\n    "))
    except Exception:
        pass

    # 额外注释（例如 spatial 子组的 Spearman 汇总）放在右上角
    if note_text:
        try:
            ax.text(
                0.98, 0.98, note_text,
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')
            )
        except Exception:
            pass
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='可视化第二轮审查特征（时间维度）')
    ap.add_argument('--config', default=str(PROJECT_ROOT / 'data/analysis/analysis_config.yaml'), help='分析配置 YAML 路径')
    ap.add_argument('--features-dir', default=str(PROJECT_ROOT / 'data/analysis/features'), help='特征 JSON 目录')
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'data/analysis/visualizations'), help='输出目录')
    ap.add_argument('--feature', default='aspect_ratio', help='特征名（默认 aspect_ratio）')
    ap.add_argument('--per-char', action='store_true', help='每字聚合（默认：关闭=样本级）')
    ap.add_argument('--agg', choices=['median', 'mean'], default='median', help='per-char 聚合方法')
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    features_dir = Path(args.features_dir)
    out_root = Path(args.output)
    feature_name = args.feature

    # 样式与颜色
    viz_cfg = cfg.get('visualization', {}) if isinstance(cfg, dict) else {}
    style_cfg = viz_cfg.get('style', {})
    color_cfg = viz_cfg.get('colors', {})
    time_colors = color_cfg.get('time_evolution') or []
    box_cfg = viz_cfg.get('boxplot', {})
    showmeans = bool(box_cfg.get('showmeans', True))
    showfliers = bool(box_cfg.get('showfliers', True))
    # 书名映射
    book_names_map = cfg.get('book_names', {}) if isinstance(cfg, dict) else {}

    # 扫描可用书籍特征
    bookname_to_path = scan_features(features_dir)
    if not bookname_to_path:
        raise RuntimeError(f'未在 {features_dir} 找到任何特征文件，请先运行 src/analysis/_archive/analyze_seg_review.py')

    def process_dimension(prefix: str, dim_cfg: Dict[str, Any], color_key: str):
        groups = dim_cfg.get('groups', []) if isinstance(dim_cfg, dict) else []
        if not groups:
            print(f'⚠️ 配置中未定义 {prefix}_dimension.groups，跳过')
            return

        palette = color_cfg.get(color_key) or []

        for gi, g in enumerate(groups, 1):
            gname = g.get('name', f'组{gi}')
            subgroups = g.get('subgroups') or []

            if subgroups:
                if prefix == 'spatial':
                    # 空间维度：按子组聚类展示（每个子组的书籍相邻排列），并画分隔线
                    subgroup_lists: List[Tuple[str, List[str]]] = []
                    for si, sg in enumerate(subgroups, 1):
                        sgname = sg.get('name', f'子组{si}')
                        resolved = resolve_books_for_subgroup(sg.get('books') or [], bookname_to_path)
                        # 子组内部按编号排序
                        def sort_key(bn: str):
                            bid = infer_book_id(bn)
                            try:
                                return (0, int(bid)) if bid is not None else (1, bn)
                            except Exception:
                                return (1, bn)
                        resolved_sorted = sorted(resolved, key=sort_key)
                        if resolved_sorted:
                            subgroup_lists.append((sgname, resolved_sorted))

                    if not subgroup_lists:
                        print(f'  ⚠️ 跳过：{gname} 无匹配书籍')
                        continue

                    labels: List[str] = []
                    values: List[List[float]] = []
                    colors_for_boxes: List[str] = []
                    color_map: Dict[str, str] = {}
                    for si, (sgname, books_in_sg) in enumerate(subgroup_lists, 1):
                        if isinstance(palette, list) and palette:
                            color_map[sgname] = palette[(si - 1) % len(palette)]
                        for bname in books_in_sg:
                            p = bookname_to_path.get(bname)
                            if not p:
                                continue
                            try:
                                data = json.loads(p.read_text(encoding='utf-8'))
                            except Exception:
                                continue
                            vals = extract_values_for_book(data, feature_name, per_char=True, agg='mean')
                            if not vals:
                                continue
                            labels.append(make_display_label(bname, book_names_map))
                            values.append(vals)
                            colors_for_boxes.append(color_map.get(sgname, '#87CEFA'))

                    # 聚类边界（在子组之间画分割线）
                    cluster_boundaries: List[int] = []
                    running = 0
                    for _, books_in_sg in subgroup_lists[:-1]:
                        running += len(books_in_sg)
                        cluster_boundaries.append(running)

                    legend_items = [(sgn, color_map.get(sgn, '#87CEFA')) for sgn, _ in subgroup_lists]
                    title = f'{gname} ({"每字" if args.per_char else "样本"}{"-"+args.agg if args.per_char else ""})'
                    fname = f'{prefix}_{safe_label(gname)}_{feature_name}_{"char" if args.per_char else "sample"}.png'
                    # 准备每个子组的 Spearman 文本摘要（μ 与 σ）
                    note_lines: List[str] = []
                    try:
                        start = 0
                        for (sgn, books_in_sg) in subgroup_lists:
                            length = len(books_in_sg)
                            end = start + length
                            x_in = list(range(1, length + 1))
                            mu_in = [float(np.mean(values[idx])) for idx in range(start, end)]
                            sd_in = [float(np.std(values[idx])) for idx in range(start, end)]
                            est_mu, lo_mu, hi_mu = _bootstrap_ci(spearman_rho, x_in, mu_in, B=300)
                            est_sd, lo_sd, hi_sd = _bootstrap_ci(spearman_rho, x_in, sd_in, B=300)
                            parts = []
                            if est_mu is not None:
                                parts.append(f"μ:{est_mu:.3f}{'' if lo_mu is None else f'[{lo_mu:.3f},{hi_mu:.3f}]'}")
                            if est_sd is not None:
                                parts.append(f"σ:{est_sd:.3f}{'' if lo_sd is None else f'[{lo_sd:.3f},{hi_sd:.3f}]'}")
                            if parts:
                                note_lines.append(f"{sgn}: " + ", ".join(parts))
                            start = end
                    except Exception:
                        pass
                    note_text = "\n".join(note_lines) if note_lines else None

                    plot_scatter_for_group(
                        gname,
                        labels,
                        values,
                        out_root / fname,
                        title,
                        ylabel={'aspect_ratio': '长宽比 (高/宽)'}.get(feature_name, feature_name),
                        colors_for_points=colors_for_boxes,
                        legend=legend_items,
                        style=style_cfg,
                        cluster_boundaries=cluster_boundaries,
                        note_text=note_text,
                    )
                    # 控制台输出组内 Spearman 摘要
                    try:
                        start = 0
                        for (sgn, books_in_sg) in subgroup_lists:
                            length = len(books_in_sg)
                            end = start + length
                            # x = 1..length（组内排序索引）
                            x_in = list(range(1, length + 1))
                            mu_in = [float(np.mean(values[idx])) for idx in range(start, end)]
                            sd_in = [float(np.std(values[idx])) for idx in range(start, end)]
                            parts = []
                            est, lo, hi = _bootstrap_ci(spearman_rho, x_in, mu_in, B=300)
                            if est is not None:
                                parts.append(f"{sgn}: ρ(μ,idx)={est:.3f}{'' if lo is None else f' CI[{lo:.3f},{hi:.3f}]'} n={len(x_in)}")
                            est, lo, hi = _bootstrap_ci(spearman_rho, x_in, sd_in, B=300)
                            if est is not None:
                                parts.append(f"ρ(σ,idx)={est:.3f}{'' if lo is None else f' CI[{lo:.3f},{hi:.3f}]'}")
                            if parts:
                                print("    " + " | ".join(parts))
                            start = end
                    except Exception:
                        pass
                    print(f'  ✓ 输出：{out_root / fname}')
                else:
                    # 其他维度：合并子组但按编号整体排序
                    book_to_sub: Dict[str, str] = {}
                    for si, sg in enumerate(subgroups, 1):
                        sgname = sg.get('name', f'子组{si}')
                        resolved = resolve_books_for_subgroup(sg.get('books') or [], bookname_to_path)
                        for bn in resolved:
                            if bn not in book_to_sub:
                                book_to_sub[bn] = sgname

                    if not book_to_sub:
                        print(f'  ⚠️ 跳过：{gname} 无匹配书籍')
                        continue

                    def sort_key(bn: str):
                        bid = infer_book_id(bn)
                        try:
                            return (0, int(bid)) if bid is not None else (1, bn)
                        except Exception:
                            return (1, bn)
                    ordered_books = sorted(book_to_sub.keys(), key=sort_key)

                    labels: List[str] = []
                    values: List[List[float]] = []
                    colors_for_boxes: List[str] = []

                    subgroup_names = []
                    color_map: Dict[str, str] = {}
                    for si, sg in enumerate(subgroups, 1):
                        sgname = sg.get('name', f'子组{si}')
                        subgroup_names.append(sgname)
                        if isinstance(palette, list) and palette:
                            color_map[sgname] = palette[(si - 1) % len(palette)]

                    for bname in ordered_books:
                        p = bookname_to_path.get(bname)
                        if not p:
                            continue
                        try:
                            data = json.loads(p.read_text(encoding='utf-8'))
                        except Exception:
                            continue
                        vals = extract_values_for_book(data, feature_name, per_char=True, agg='mean')
                        if not vals:
                            continue
                        labels.append(make_display_label(bname, book_names_map))
                        values.append(vals)
                        colors_for_boxes.append(color_map.get(book_to_sub.get(bname), '#87CEFA'))

                    if not labels:
                        print(f'  ⚠️ 跳过：{gname} 无可绘制数据')
                        continue

                    legend_items = [(sgn, color_map.get(sgn, '#87CEFA')) for sgn in subgroup_names]
                    title = f'{gname} ({"每字" if args.per_char else "样本"}{"-"+args.agg if args.per_char else ""})'
                    fname = f'{prefix}_{safe_label(gname)}_{feature_name}_{"char" if args.per_char else "sample"}.png'
                    plot_scatter_for_group(
                        gname,
                        labels,
                        values,
                        out_root / fname,
                        title,
                        ylabel={'aspect_ratio': '长宽比 (高/宽)'}.get(feature_name, feature_name),
                        colors_for_points=colors_for_boxes,
                        legend=legend_items,
                        style=style_cfg,
                    )
                    print(f'  ✓ 输出：{out_root / fname}')
            else:
                books_spec = g.get('books') or []
                resolved_books = resolve_books_for_subgroup(books_spec, bookname_to_path)
                if not resolved_books:
                    print(f'  ⚠️ 跳过：{gname} 无匹配书籍')
                    continue

                # 按编号排序
                def sort_key(bn: str):
                    bid = infer_book_id(bn)
                    try:
                        return (0, int(bid)) if bid is not None else (1, bn)
                    except Exception:
                        return (1, bn)
                ordered_books = sorted(resolved_books, key=sort_key)

                labels: List[str] = []
                values: List[List[float]] = []
                for bname in ordered_books:
                    p = bookname_to_path.get(bname)
                    if not p:
                        continue
                    try:
                        data = json.loads(p.read_text(encoding='utf-8'))
                    except Exception:
                        continue
                    # 每字平均值
                    vals = extract_values_for_book(data, feature_name, per_char=True, agg='mean')
                    if not vals:
                        continue
                    labels.append(make_display_label(bname, book_names_map))
                    values.append(vals)
                if not labels:
                    print(f'  ⚠️ 跳过：{gname} 无可绘制数据')
                    continue

                title = f'{gname} ({"每字" if args.per_char else "样本"}{"-"+args.agg if args.per_char else ""})'
                fname = f'{prefix}_{safe_label(gname)}_{feature_name}_{"char" if args.per_char else "sample"}.png'
                # 无子组：不区分颜色，也不显示 legend（按需求简化）
                legend_items = None
                colors_for_boxes = None
                plot_scatter_for_group(
                    gname,
                    labels,
                    values,
                    out_root / fname,
                    title,
                    ylabel={'aspect_ratio': '长宽比 (高/宽)'}.get(feature_name, feature_name),
                    colors_for_points=colors_for_boxes,
                    legend=legend_items,
                    style=style_cfg,
                )
                print(f'  ✓ 输出：{out_root / fname}')

    # 运行三个维度（时间/空间/类型）
    process_dimension('time', cfg.get('time_dimension', {}), 'time_evolution')
    process_dimension('spatial', cfg.get('spatial_dimension', {}), 'spatial_region')
    process_dimension('type', cfg.get('type_dimension', {}), 'book_type')


if __name__ == '__main__':
    main()
