from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official LOBSTER AAPL level-10 public sample.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default="data/raw/aapl_level10")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "lobster_sample.zip"

    request = urllib.request.Request(args.url, headers={"User-Agent": "lob-research-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)
    archive.unlink()

    messages = sorted(output_dir.rglob("*message*.csv"))
    books = sorted(output_dir.rglob("*orderbook*.csv"))
    if len(messages) != 1 or len(books) != 1:
        raise RuntimeError(
            f"Expected one message and one orderbook CSV, found {len(messages)} and {len(books)}"
        )

    print(f"messages={messages[0]}")
    print(f"book={books[0]}")


if __name__ == "__main__":
    main()
