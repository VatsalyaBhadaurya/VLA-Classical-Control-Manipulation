"""Put the repo root on sys.path so `scripts/*.py` can import the package
without an editable install. Import this first in each script."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
