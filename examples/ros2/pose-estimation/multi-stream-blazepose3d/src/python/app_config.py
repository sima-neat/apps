"""JSON conversion for numerical results; no inference or ROS dispatch."""
from dataclasses import asdict, is_dataclass
from pathlib import Path

def json_value(value):
    """Convert native result values without rounding or changing coordinates."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported result value: {type(value).__name__}")
