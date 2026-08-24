"""日志工具：统一配置 logger，带时间戳/模块名/级别。

用法:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("消息 %s", param)
"""
import logging
import sys


def get_logger(name: str = "rag", level: str = "INFO") -> logging.Logger:
    """获取配置好的 logger。

    首次调用时配置 handler 和 formatter，后续调用复用同一 logger。
    propagate=False 避免根 logger 重复打印。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
