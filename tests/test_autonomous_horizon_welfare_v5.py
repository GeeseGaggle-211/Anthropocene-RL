"""Trade-good slots must not silently turn cloth observations into food evidence."""

import importlib.util
import sys
from pathlib import Path


def test_food_stock_uses_native_food_slot_with_distinct_cloth_and_food():
    directory = Path("scripts").resolve()
    spec = importlib.util.spec_from_file_location(
        "horizon_welfare", directory / "evaluate_autonomous_horizon_v5.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    assert module.food_stock(dict(Stocks=[91, 29, 13, 17, 0, 0, 0, 0, 0, 0, 0])) == 17
