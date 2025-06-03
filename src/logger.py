import logging
import os
from datetime import datetime
import pytz


def configure_logger():
    logs_path = "artifacts/logs"
    os.makedirs(logs_path, exist_ok=True)
    date_format = "%d-%m-%Y-%H-%M-%S"

    # Get Vietnam timezone
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vietnam_time = datetime.now(pytz.utc).astimezone(vietnam_tz)

    # Create log file with Vietnam time
    log_file = f"{vietnam_time.strftime(date_format)}.log"
    log_file_path = os.path.join(logs_path, log_file)

    # Custom converter for formatter
    def vietnam_time_converter(*args):
        return datetime.now(pytz.utc).astimezone(vietnam_tz).timetuple()

    # Apply converter
    logging.Formatter.converter = vietnam_time_converter

    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s\n %(message)s",
        level=logging.INFO,
        force=True,
    )

    return log_file_path
