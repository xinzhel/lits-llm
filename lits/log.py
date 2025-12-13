import os
import logging
from typing import Iterable, Tuple

class _NamespaceFilter(logging.Filter):
    """Keeps the log file scoped to the project namespaces."""

    def __init__(self, namespaces: Iterable[str] | None):
        super().__init__()
        self.namespaces: Tuple[str, ...] = tuple(ns for ns in (namespaces or ()) if ns)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.namespaces:
            return True
        if record.name in {"root", "__main__"}:
            return True
        return record.name.startswith(self.namespaces)


def setup_logging(
    run_id,
    result_dir,
    add_console_handler=False,
    verbose=False,
    allowed_namespaces=("lits", "mem0"),
):
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{run_id}.log")
    namespace_filter = _NamespaceFilter(allowed_namespaces)

    # ～～～～～～ 文件处理器 (Begin) ～～～～～～
    # Create file handler. If the file exists, append the log.
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    
    # 文件处理器的级别设置为 logging.DEBUG，这意味着它会接收所有日志事件（只要记录器本身的级别允许）
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    if namespace_filter.namespaces:
        file_handler.addFilter(namespace_filter)
    # ～～～～～～ 文件处理器 (End) ～～～～～～
    
    # ～～～～～～ Logger Setup (Begin) ～～～～～～
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO) # # logging.DEBUG（会记录所有级别的消息）; logging.INFO（会记录 INFO、WARNING、ERROR、CRITICAL 级别的消息）
    logger.propagate = False # 没有实际效果，因为没有“上一级”可以继续冒泡
    logger.handlers.clear() # 清除根记录器上所有现有的处理器（Handler）。这是最关键的一步，确保每次调用该函数时，日志配置都是全新的，避免重复或冲突。
    logger.addHandler(file_handler) # Add handlers to logger
    # ～～～～～～ Logger Setup (End) ～～～～～～

    #  ～～～～～～  Console handler (Begin) ～～～～～～
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        if namespace_filter.namespaces:
            console_handler.addFilter(namespace_filter)
        logger.addHandler(console_handler)
    #  ～～～～～～  Console handler (End) ～～～～～～
    return logger
