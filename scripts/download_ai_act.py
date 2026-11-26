"""Download the AI Act (Regulation (EU) 2024/1689) from EUR-Lex.

Usage:
    uv run python scripts/download_ai_act.py                 # EUR-Lex / Cellar
    uv run python scripts/download_ai_act.py --mirror pypi   # hash-pinned PyPI mirror

The file is written to data/raw/ (git-ignored: the text is downloaded, not versioned).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ai_act_rag.infrastructure.eurlex import (
    download_ai_act,
    download_from_pypi_mirror,
    parse_ai_act,
    sha256,
)

DEFAULT_DEST = Path("data/raw/ai_act_32024R1689_en.xhtml")


def main() -> int:
    """Run the download and print a structural summary.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--mirror",
        choices=["eurlex", "pypi", "auto"],
        default="auto",
        help="auto = try EUR-Lex first, then the hash-pinned PyPI mirror",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mirror in ("eurlex", "auto"):
        try:
            download_ai_act(args.dest)
        except RuntimeError as exc:
            if args.mirror == "eurlex":
                logging.error("%s", exc)
                return 1
            logging.warning("EUR-Lex unreachable, using the PyPI mirror")
            download_from_pypi_mirror(args.dest)
    else:
        download_from_pypi_mirror(args.dest)

    units = parse_ai_act(args.dest.read_bytes())
    counts = {t: sum(u.unit_type == t for u in units) for t in ("article", "recital", "annex")}
    print(f"Saved {args.dest} sha256={sha256(args.dest)} units={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
