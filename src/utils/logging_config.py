import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    level=logging.INFO, log_file=None, log_dir="logs", json_format=False, include_process_info=False
):
    """
    Configure logging for the application

    Args:
        level: Logging level
        log_file: Optional log file path
        log_dir: Directory for log files
        json_format: Use JSON formatting
        include_process_info: Include process/thread info
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if json_format:
        formatter = JsonFormatter(include_process_info=include_process_info)
    else:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if include_process_info:
            format_string = (
                "%(asctime)s - %(process)d - %(thread)d - %(name)s - %(levelname)s - %(message)s"
            )
        formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_path = Path(log_dir) / log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = Path(log_dir) / f"crosssell_{timestamp}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(level)

    # Log startup message
    root_logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")

    return root_logger


class JsonFormatter(logging.Formatter):
    """JSON log formatter"""

    def __init__(self, include_process_info=False):
        super().__init__()
        self.include_process_info = include_process_info

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if self.include_process_info:
            log_data.update(
                {
                    "process": record.process,
                    "thread": record.thread,
                    "thread_name": record.threadName,
                }
            )

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)
