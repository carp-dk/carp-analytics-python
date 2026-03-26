"""Command-line entrypoint for CARP Analytics."""

from __future__ import annotations

import sys

from carp.commandline.app import main

if __name__ == "__main__":
    sys.exit(main())
