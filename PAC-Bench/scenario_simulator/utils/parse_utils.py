from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))

def parse_interval(range_expr: str, *, max_value: int | None = None) -> List[int]:
    expr = range_expr.strip().replace(" ", "")
    if len(expr) < 5:
        raise ValueError("range must follow interval notation like '[0,10)'.")

    start_bracket, end_bracket = expr[0], expr[-1]
    if start_bracket not in "[(" or end_bracket not in ")]":
        raise ValueError("range must start with '[' or '(' and end with ']' or ')'.")

    bounds = expr[1:-1].split(",")
    if len(bounds) != 2:
        raise ValueError("range must contain exactly two bounds separated by a comma.")

    try:
        raw_start = int(bounds[0])
        raw_end = int(bounds[1])
    except ValueError as exc:
        raise ValueError("range bounds must be integers.") from exc

    start = raw_start if start_bracket == "[" else raw_start + 1
    stop = raw_end + 1 if end_bracket == "]" else raw_end

    if start >= stop:
        raise ValueError("range does not contain any valid values.")

    if start < 0:
        raise ValueError("range values must be non-negative.")

    if max_value is not None and stop > max_value:
        raise ValueError(f"range cannot exceed max_value ({max_value}).")

    return list(range(start, stop))


def parse_range(range_expr: str) -> List[int]:
    return parse_interval(range_expr, max_value=MAX_SESSIONS)