"""
进度管理模块

用于记录和恢复批处理进度，支持断点续传
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

# 导入项目根目录
from src.config import PROJECT_ROOT


def _get_relative_path(path: str) -> str:
    """
    将绝对路径转换为相对于项目根目录的路径

    Args:
        path: 绝对路径

    Returns:
        相对路径（如果在项目内）或原路径
    """
    try:
        path_obj = Path(path).resolve()
        project_root = Path(PROJECT_ROOT).resolve()
        return str(path_obj.relative_to(project_root))
    except (ValueError, AttributeError):
        # 如果路径不在项目内，或转换失败，返回原路径
        return str(path)


def _resolve_path(rel_path: str) -> str:
    """
    将项目相对路径转换为绝对路径

    Args:
        rel_path: 相对路径

    Returns:
        绝对路径
    """
    if os.path.isabs(rel_path):
        return rel_path
    return str(Path(PROJECT_ROOT) / rel_path)


class ProgressTracker:
    """
    进度跟踪器

    使用 progress.json 文件记录处理进度，支持断点续传
    """

    def __init__(self, progress_file: str, stage: str):
        """
        初始化进度跟踪器

        Args:
            progress_file: progress.json 文件路径
            stage: 处理阶段名称（如 'preprocess', 'ocr'）
        """
        self.progress_file = progress_file
        self.stage = stage
        self.data = self._load()
        self._input_dir_abs = None  # 保存输入目录的绝对路径（用于路径转换）

    def _load(self) -> Dict:
        """加载进度文件"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # 文件损坏，返回空数据
                return {}
        return {}

    def _save(self):
        """保存进度文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _get_stage_data(self) -> Dict:
        """获取当前阶段的数据"""
        if self.stage not in self.data:
            self.data[self.stage] = {
                'input_dir': None,
                'output_dir': None,
                'completed': [],
                'failed': [],
                'last_update': None,
            }
        return self.data[self.stage]

    def init_session(self, input_dir: str, output_dir: str, force: bool = False):
        """
        初始化处理会话

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            force: 是否强制重新处理（清除进度）
        """
        stage_data = self._get_stage_data()

        # 保存绝对路径（用于后续路径转换）
        self._input_dir_abs = str(Path(input_dir).resolve())

        # 转换为相对路径（相对于项目根目录）
        input_dir_rel = _get_relative_path(str(input_dir))
        output_dir_rel = _get_relative_path(str(output_dir))

        # 如果是新的输入/输出目录，或强制重新处理，则重置进度
        if force or stage_data['input_dir'] != input_dir_rel or stage_data['output_dir'] != output_dir_rel:
            stage_data['input_dir'] = input_dir_rel
            stage_data['output_dir'] = output_dir_rel
            stage_data['completed'] = []
            stage_data['failed'] = []
            stage_data['last_update'] = datetime.now().isoformat()
            self._save()

    def _to_project_relative_path(self, filename: str) -> str:
        """
        将相对于 input_dir 的路径转换为相对于项目根目录的路径

        Args:
            filename: 相对于 input_dir 的文件路径

        Returns:
            相对于项目根目录的路径
        """
        if not self._input_dir_abs:
            return filename

        # 拼接为绝对路径
        abs_path = os.path.join(self._input_dir_abs, filename)
        # 转换为项目相对路径
        return _get_relative_path(abs_path)

    def _from_project_relative_path(self, project_rel_path: str) -> str:
        """
        将相对于项目根目录的路径转换回相对于 input_dir 的路径

        Args:
            project_rel_path: 相对于项目根目录的路径

        Returns:
            相对于 input_dir 的路径
        """
        if not self._input_dir_abs:
            return project_rel_path

        # 转换为绝对路径
        abs_path = _resolve_path(project_rel_path)
        # 计算相对于 input_dir 的路径
        try:
            return str(Path(abs_path).relative_to(self._input_dir_abs))
        except ValueError:
            return project_rel_path

    def is_completed(self, filename: str) -> bool:
        """
        检查文件是否已成功处理

        Args:
            filename: 相对于 input_dir 的文件路径
        """
        stage_data = self._get_stage_data()
        # 转换为项目相对路径后比较
        project_rel_path = self._to_project_relative_path(filename)
        return project_rel_path in stage_data.get('completed', [])

    def is_failed(self, filename: str) -> bool:
        """
        检查文件是否处理失败过

        Args:
            filename: 相对于 input_dir 的文件路径
        """
        stage_data = self._get_stage_data()
        # 转换为项目相对路径后比较
        project_rel_path = self._to_project_relative_path(filename)
        return project_rel_path in stage_data.get('failed', [])

    def mark_completed(self, filename: str):
        """
        标记文件为已完成

        Args:
            filename: 相对于 input_dir 的文件路径
        """
        stage_data = self._get_stage_data()

        # 转换为项目相对路径
        project_rel_path = self._to_project_relative_path(filename)

        # 从失败列表中移除（如果存在）
        if project_rel_path in stage_data.get('failed', []):
            stage_data['failed'].remove(project_rel_path)

        # 添加到完成列表（避免重复）
        if project_rel_path not in stage_data.get('completed', []):
            stage_data['completed'].append(project_rel_path)

        stage_data['last_update'] = datetime.now().isoformat()
        self._save()

    def mark_failed(self, filename: str):
        """
        标记文件为处理失败

        Args:
            filename: 相对于 input_dir 的文件路径
        """
        stage_data = self._get_stage_data()

        # 转换为项目相对路径
        project_rel_path = self._to_project_relative_path(filename)

        # 从完成列表中移除（如果存在）
        if project_rel_path in stage_data.get('completed', []):
            stage_data['completed'].remove(project_rel_path)

        # 添加到失败列表（避免重复）
        if project_rel_path not in stage_data.get('failed', []):
            stage_data['failed'].append(project_rel_path)

        stage_data['last_update'] = datetime.now().isoformat()
        self._save()

    def get_completed_files(self) -> List[str]:
        """获取已完成的文件列表"""
        stage_data = self._get_stage_data()
        return stage_data.get('completed', [])

    def get_failed_files(self) -> List[str]:
        """获取失败的文件列表"""
        stage_data = self._get_stage_data()
        return stage_data.get('failed', [])

    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """
        获取待处理的文件列表

        Args:
            all_files: 所有文件列表（相对于 input_dir 的路径）

        Returns:
            待处理的文件列表（未完成 + 失败的，相对于 input_dir 的路径）
        """
        stage_data = self._get_stage_data()
        completed_project_paths = set(stage_data.get('completed', []))

        # 筛选未完成的文件
        pending = []
        for f in all_files:
            project_rel_path = self._to_project_relative_path(f)
            if project_rel_path not in completed_project_paths:
                pending.append(f)

        return pending

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stage_data = self._get_stage_data()
        return {
            'completed': len(stage_data.get('completed', [])),
            'failed': len(stage_data.get('failed', [])),
            'last_update': stage_data.get('last_update'),
        }

    def clear(self):
        """清除当前阶段的进度"""
        if self.stage in self.data:
            del self.data[self.stage]
            self._save()


def get_default_progress_file(base_dir: str) -> str:
    """
    获取默认的 progress.json 文件路径

    Args:
        base_dir: 基础目录（通常是输出目录）

    Returns:
        progress.json 文件路径
    """
    return os.path.join(base_dir, '.progress.json')
