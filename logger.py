import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        """
        Initialize the Logger instance. Creates a logs directory and log file if not present.
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # Optional: Logs to console as well
            ]
        )

    @staticmethod
    def log_info(message):
        """Log an info message."""
        logging.info(message)

    @staticmethod
    def log_error(message):
        """Log an error message."""
        logging.error(message)

    @staticmethod
    def log_debug(message):
        """Log a debug message."""
        logging.debug(message)

    @staticmethod
    def log_warning(message):
        """Log a warning message."""
        logging.warning(message)
