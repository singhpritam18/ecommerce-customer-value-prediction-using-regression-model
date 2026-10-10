"""
conftest.py — pytest root configuration.

Inserts `src/` onto sys.path as the `ecommerce_customer_value_prediction` namespace
so that `import ecommerce_customer_value_prediction.xxx` resolves to `src/xxx`
without requiring a special build-system mapping.
"""

import sys
from pathlib import Path

# Make `src/` importable as the `ecommerce_customer_value_prediction` package.
# This is equivalent to:  PYTHONPATH=.  when src/__init__.py exists and the
# installed editable wheel maps "src" → "ecommerce_customer_value_prediction".
sys.path.insert(0, str(Path(__file__).parent))

import importlib
import types

# Create a top-level alias so that:
#   from ecommerce_customer_value_prediction.data import ... works
# by pointing the ecommerce_customer_value_prediction namespace at the src/ package.
_src_path = Path(__file__).parent / "src"
if _src_path not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(_src_path))

# Register ecommerce_customer_value_prediction as an alias for the src package
import src as _src_pkg  # noqa: E402

_alias = "ecommerce_customer_value_prediction"
if _alias not in sys.modules:
    sys.modules[_alias] = _src_pkg
    # Also register all sub-modules already loaded under src.*
    for key in list(sys.modules):
        if key.startswith("src."):
            alias_key = _alias + key[len("src"):]
            if alias_key not in sys.modules:
                sys.modules[alias_key] = sys.modules[key]
