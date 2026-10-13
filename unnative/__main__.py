"""Enable `python3 -m unnative ...` with no install step."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
