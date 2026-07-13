import ast
from pathlib import Path

from qpiai_quantum.config import HTTP_TIMEOUT


def test_http_timeout_has_bounded_connect_and_read_values():
    assert HTTP_TIMEOUT == (10, 120)


def test_all_requests_calls_specify_a_timeout():
    package_dir = Path(__file__).parents[1] / "qpiai_quantum"

    for source_file in package_dir.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if not isinstance(call.func.value, ast.Name) or call.func.value.id != "requests":
                continue
            assert any(keyword.arg == "timeout" for keyword in call.keywords), (
                f"{source_file.relative_to(package_dir.parent)}:{call.lineno} "
                "is missing a request timeout"
            )
