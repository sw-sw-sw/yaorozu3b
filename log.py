import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from datetime import datetime

DRAW_INFO = 15
logging.addLevelName(DRAW_INFO, "DRAW_INFO")
class ColoredFormatter(logging.Formatter):
    """カラーコードを使用してログメッセージをフォーマットするクラス"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',  # シアン
        'INFO': '\033[0;32m',   # 緑
        'WARNING': '\033[0;33m',  # 黄
        'ERROR': '\033[0;31m',  # 赤
        'CRITICAL': '\033[0;35m',  # マゼンタ
        'RESET': '\033[0m',  # リセット
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"

class SimulationLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.logger = logging.getLogger('SimulationLogger')
        self.logger.setLevel(logging.NOTSET)  # 最低レベルをDEBUGに設定

        self.setup_handlers()

    def setup_handlers(self):
        # メインログファイルのハンドラ
        main_log_path = os.path.join(self.log_dir, 'simulation.log')
        main_file_handler = RotatingFileHandler(main_log_path, maxBytes=10*1024*1024, backupCount=5)
        main_file_handler.setLevel(logging.NOTSET)
        main_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(main_file_handler)

        # エラーログファイルのハンドラ
        error_log_path = os.path.join(self.log_dir, 'error.log')
        error_file_handler = TimedRotatingFileHandler(error_log_path, when='midnight', interval=1, backupCount=7)
        error_file_handler.setLevel(logging.NOTSET)  # WARNING以上のみ
        error_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(error_file_handler)

        # コンソール出力用のハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.NOTSET)
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

    def set_log_level(self, level):
        """ログレベルを動的に変更するメソッド"""
        self.logger.setLevel(level)

    def debug(self, message):
        self.logger.debug(message)
            


    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def error(self, message):
        self.logger.error(f"ERROR: {message}")

    def critical(self, message):
        self.logger.critical(f"CRITICAL: {message}")

    def exception(self, message):
        self.logger.exception(f"EXCEPTION: {message}")

# グローバルなロガーオブジェクトの作成
simulation_logger = SimulationLogger()

# 他のモジュールからインポートして使用する関数
def get_logger():
    return simulation_logger

# ログレベルを変更する関数
def set_log_level(level):
    simulation_logger.set_log_level(level)