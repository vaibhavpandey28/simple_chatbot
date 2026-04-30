import logging
import os
import sys


_CONFIGURED = False


def _is_logger_enabled() -> bool:
    value = os.getenv("LOGGER", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_color_enabled() -> bool:
    value = os.getenv("LOGGER_COLOR", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname, "")
        if color:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
        message = super().format(record)
        record.levelname = original_levelname
        return message


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    if not _is_logger_enabled():
        logging.disable(logging.CRITICAL)
        _CONFIGURED = True
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    formatter = ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter if _is_color_enabled() else logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)
