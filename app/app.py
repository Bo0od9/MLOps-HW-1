import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scorer import make_pred


def load_cfg():
    for p in ["configs/config.yaml", "config/config.yaml"]:
        cfg_path = PROJECT_ROOT / p
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {
        "paths": {"input_dir": "input", "output_dir": "output"},
        "data": {"output_file_name": "sample_submission.csv"},
        "inference": {"threshold": 0.5, "produce_feature_importances": True, "produce_density_plot": True},
    }


LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
fh = RotatingFileHandler(LOGS_DIR / "service.log", maxBytes=5 * (2**20), backupCount=2, encoding="utf-8")
sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format=fmt, handlers=[fh, sh], force=True)
logger = logging.getLogger("ml_service")


class ProcessingService:
    def __init__(self):
        self.cfg = load_cfg()
        self.input_dir = Path(self.cfg["paths"].get("input_dir", "input")).resolve()
        self.output_dir = Path(self.cfg["paths"].get("output_dir", "output")).resolve()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Service initialized. Watching input_dir=%s, output_dir=%s", self.input_dir, self.output_dir)

    def _sanity_check(self, file_path: Path) -> None:
        try:
            pd.read_csv(file_path, nrows=1)
        except Exception as e:
            raise ValueError(f"Invalid CSV: {file_path} ({e})")

    def process_file(self, file_path: Path) -> None:
        try:
            if not file_path.exists():
                logger.info("File disappeared before processing: %s", file_path)
                return

            time.sleep(2)
            self._sanity_check(file_path)
            test_csv = self.input_dir / "test.csv"
            test_csv.write_bytes(file_path.read_bytes())
            logger.info("Copied %s -> %s", file_path.name, test_csv)

            logger.info("Calling scorer.make_pred()")
            make_pred()
            logger.info("make_pred() finished for %s", file_path.name)

        except Exception as e:
            logger.error("Error during processing %s: %s", file_path, e, exc_info=True)


class FileHandler(FileSystemEventHandler):
    def __init__(self, service: ProcessingService):
        super().__init__()
        self.service = service
        self._last_processed = None

    def _maybe_process(self, path: Path):
        if path.is_dir() or path.suffix.lower() != ".csv":
            return
        if path.name != "test.csv":
            return
        if self._last_processed == path:
            return

        self._last_processed = path
        logger.info(f"Detected file: {path}")
        self.service.process_file(path)

    def on_created(self, event):
        self._maybe_process(Path(event.src_path))

    def on_modified(self, event):
        self._maybe_process(Path(event.src_path))

    def on_moved(self, event):
        self._maybe_process(Path(event.dest_path))


def main():
    logger.info("Starting ML scoring service...")
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=str(service.input_dir), recursive=False)
    observer.start()
    logger.info("File observer started on %s", service.input_dir)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
