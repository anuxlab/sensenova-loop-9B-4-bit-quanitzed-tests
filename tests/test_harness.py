import json
from sensenova_eval.runner import extract_code, run_code_tests
from sensenova_eval.tasks import TASKS


def test_task_catalog_has_expected_tracks():
    assert {"basic", "reasoning", "coding", "agentic", "longcontext"}.issubset(TASKS)


def test_extract_code_fence():
    assert extract_code("```python\ndef f(): pass\n```").startswith("def f")


def test_code_test_runner_passes():
    ok, _ = run_code_tests("def add(a, b): return a + b", [{"input": "assert add(2, 3) == 5"}])
    assert ok


def test_code_test_runner_fails_bad_candidate():
    ok, _ = run_code_tests("def add(a, b): return 0", [{"input": "assert add(2, 3) == 5"}])
    assert not ok
