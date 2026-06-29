"""Step 14 - data provenance.

Records the pinned GlioVis source commit and SHA-256 checksums of all raw input
files, written to data/dictionary/data_manifest.json for archival reproducibility.
"""
import os, sys, pathlib, hashlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import requests
from src import config
from src.utils import get_logger

log = get_logger()


def provenance():
    sha = "unknown"
    try:
        r = requests.get("https://api.github.com/repos/msquatrito/shiny_GlioVis/commits/master", timeout=30)
        sha = r.json().get("sha", "unknown")
    except Exception as e:
        log.warning("commit lookup failed: %s", e)
    manifest = {"gliovis_repo": "msquatrito/shiny_GlioVis",
                "gliovis_commit_pinned": getattr(config, "GLIOVIS_COMMIT", "eae2ce7852ba"),
                "gliovis_commit_latest_master": sha, "files": {}}
    for p in sorted(config.RAW.glob("*")):
        if p.is_file():
            manifest["files"][p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                         "bytes": p.stat().st_size}
    json.dump(manifest, open(config.DICTIONARY / "data_manifest.json", "w"), indent=2)
    log.info("wrote data_manifest.json (%d files, pinned commit %s)",
             len(manifest["files"]), manifest["gliovis_commit_pinned"])
    return manifest


if __name__ == "__main__":
    m = provenance()
    print("PROVENANCE pinned commit:", m["gliovis_commit_pinned"], "files:", len(m["files"]))
