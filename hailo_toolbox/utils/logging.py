"""
Logging module that provides unified logging functionality for the entire project.

This module provides the following features:
1. Support for multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
2. Support for simultaneous output to console and file
3. Support for custom log formats
4. Support for log file rotation
5. Support for shared log configuration across multiple modules
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional, Union, List, Tuple
import datetime
import atexit


# Global dictionary to store created loggers
_loggers = {}

# Default configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = None
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5


def setup_logger(
    name: str,
    level: Union[int, str] = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    console: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    propagate: bool = False,
) -> logging.Logger:
    """
    Set up a logger.

    Args:
        name: Logger name, usually use module name
        level: Log level, can be number or string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format
        date_format: Date format
        log_file: Log file name, if None then auto-generate (based on name)
        log_dir: Log file directory
        console: Whether to output to console
        max_bytes: Maximum bytes per log file, rotate when exceeded
        backup_count: Number of backup files to keep
        propagate: Whether to pass logs to parent logger

    Returns:
        Configured logger
    """
    # Handle log level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    # 检查记录器是否已存在
    if name in _loggers:
        return _loggers[name]

    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(log_format, date_format)

    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # 添加文件处理器（如果需要）
    if log_file is not None or log_dir is not None:
        # 如果log_file为None，则根据name生成
        if log_file is None:
            safe_name = name.replace(".", "_")
            log_file = f"{safe_name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, log_file)

        # 创建轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # 存储记录器以便复用
    _loggers[name] = logger

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的日志记录器，如果不存在则创建一个基础配置的记录器。

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    if name in _loggers:
        return _loggers[name]
    else:
        # 创建一个基础配置的记录器
        return setup_logger(name)


def flush_all_loggers() -> None:
    """
    强制刷新所有日志记录器的输出流。
    通常在程序退出前调用，确保所有日志都被写入文件。
    """
    for name, logger in _loggers.items():
        for handler in logger.handlers:
            handler.flush()


# 注册应用退出时刷新所有日志
atexit.register(flush_all_loggers)


# 创建根记录器
# root_logger = setup_logger("hailo_toolbox", console=True, log_file="hailo_toolbox.log")
