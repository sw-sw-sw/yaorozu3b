import logging, sys, psutil, time

def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

logger = setup_logging()


def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"CPU usage: {cpu_percent}%, Memory usage: {memory_percent}%")
        time.sleep(5)

