'''
使用方法
# クラス内での使用
class MyClass:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def some_method(self):
        self.logger.info("some_methodが呼び出されました。")

# クラス外での使用（例：utils.py）
from log import get_logger

logger = get_logger(__name__)

def some_function():
    logger.info("some_functionが呼び出されました。")

# メインスクリプトでの使用（例：main.py）
if __name__ == "__main__":
    logger = get_logger("Main")
    logger.info("メインスクリプトが開始されました。")
'''

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from datetime import datetime

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

class ClassNameFilter(logging.Filter):
    """クラス名をログレコードに追加するフィルター"""
    def __init__(self, class_name):
        super().__init__()
        self.class_name = class_name

    def filter(self, record):
        record.classname = self.class_name
        return True

class SimulationLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.logger = logging.getLogger('SimulationLogger')
        self.logger.setLevel(logging.DEBUG)

        self.setup_handlers()

    def setup_handlers(self):
        # メインログファイルのハンドラ
        main_log_path = os.path.join(self.log_dir, 'simulation.log')
        main_file_handler = RotatingFileHandler(main_log_path, maxBytes=10*1024*1024, backupCount=5)
        main_file_handler.setLevel(logging.DEBUG)
        main_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(classname)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(main_file_handler)

        # エラーログファイルのハンドラ
        error_log_path = os.path.join(self.log_dir, 'error.log')
        error_file_handler = TimedRotatingFileHandler(error_log_path, when='midnight', interval=1, backupCount=7)
        error_file_handler.setLevel(logging.WARNING)  # WARNING以上のみ
        error_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(classname)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(error_file_handler)

        # コンソール出力用のハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(classname)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

    def get_logger(self, name):
        """名前付きのロガーを取得するメソッド"""
        logger = logging.LoggerAdapter(self.logger, extra={'classname': name})
        logger.logger.addFilter(ClassNameFilter(name))
        return logger

    def set_log_level(self, level):
        """ログレベルを動的に変更するメソッド"""
        self.logger.setLevel(level)

# グローバルなロガーオブジェクトの作成
simulation_logger = SimulationLogger()

# 他のモジュールからインポートして使用する関数
def get_logger(name=None):
    if name is None:
        # 呼び出し元のモジュール名を取得
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'Unknown')
    return simulation_logger.get_logger(name)

# ログレベルを変更する関数
def set_log_level(level):
    simulation_logger.set_log_level(level)