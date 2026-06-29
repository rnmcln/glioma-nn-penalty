"""Step 01 - download all public data (GlioVis .Rds, cBioPortal TCGA, GEO matrices)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_download import download_all
from src.utils import get_logger

if __name__ == "__main__":
    log = get_logger()
    download_all(force="--force" in sys.argv)
    log.info("Download complete. Raw files in data/raw/")
