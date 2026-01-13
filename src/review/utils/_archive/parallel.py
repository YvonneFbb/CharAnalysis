"""
并发处理工具模块

提供多线程/多进程处理功能，加速 PDF转换、预处理、OCR 等任务
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from typing import Callable, List, Any, Tuple, Optional


class ParallelProcessor:
    """并发处理器 - 支持多线程和多进程"""

    def __init__(self, workers: int = 4, mode: str = 'thread'):
        """
        初始化并发处理器

        Args:
            workers: 并发工作线程/进程数
            mode: 'thread' 或 'process'
        """
        self.workers = workers
        self.mode = mode
        self.lock = threading.Lock()  # 用于线程安全的进度更新

    def process_items(self,
                     items: List[Any],
                     process_func: Callable,
                     desc: str = "处理进度",
                     return_results: bool = False) -> Tuple[int, int, Optional[List]]:
        """
        并发处理多个项目

        Args:
            items: 待处理的项目列表
            process_func: 处理函数，接收单个项目，返回 (success: bool, result: Any)
            desc: 进度条描述
            return_results: 是否返回所有结果

        Returns:
            (成功数量, 总数量, 结果列表或None)
        """
        if not items:
            return 0, 0, [] if return_results else None

        # 选择执行器
        if self.mode == 'process':
            ExecutorClass = ProcessPoolExecutor
        else:
            ExecutorClass = ThreadPoolExecutor

        success_count = 0
        total_count = len(items)
        results = [] if return_results else None

        # 使用 ThreadPoolExecutor/ProcessPoolExecutor 并发处理
        with ExecutorClass(max_workers=self.workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_func, item): item
                for item in items
            }

            # 使用 tqdm 显示进度
            with tqdm(total=total_count, desc=desc, unit="file") as pbar:
                for future in as_completed(future_to_item):
                    try:
                        success, result = future.result()
                        if success:
                            with self.lock:
                                success_count += 1
                        if return_results:
                            with self.lock:
                                results.append(result)
                    except Exception as e:
                        tqdm.write(f"✗ 处理失败: {str(e)}")
                    finally:
                        pbar.update(1)

        return success_count, total_count, results


def process_with_threads(items: List[Any],
                        process_func: Callable,
                        workers: int = 4,
                        desc: str = "处理进度") -> Tuple[int, int]:
    """
    使用多线程处理项目列表（快捷函数）

    Args:
        items: 待处理的项目列表
        process_func: 处理函数，接收单个项目，返回 bool 表示成功/失败
        workers: 线程数
        desc: 进度条描述

    Returns:
        (成功数量, 总数量)
    """
    processor = ParallelProcessor(workers=workers, mode='thread')

    # 包装处理函数以返回 (success, None) 格式
    def wrapped_func(item):
        success = process_func(item)
        return (success, None)

    success_count, total_count, _ = processor.process_items(
        items, wrapped_func, desc=desc, return_results=False
    )
    return success_count, total_count
