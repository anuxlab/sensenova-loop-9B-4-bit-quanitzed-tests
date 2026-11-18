from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .backend import load_backend
from .tasks import TASKS


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def run_code_tests(code: str, tests: list[dict[str, str]]) -> tuple[bool, str]:
    # Generated code is untrusted. This opt-in runner limits time and resource use,
    # but is not a security boundary; use a disposable container for hostile code.
    script = code + "\n\n" + "\n".join(t["input"] for t in tests)
    with tempfile.TemporaryDirectory(prefix="sensenova-code-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)], cwd=tmp,
                capture_output=True, text=True, timeout=8,
                env={"PYTHONIOENCODING": "utf-8", "PATH": ""},
            )
            return proc.returncode == 0, (proc.stderr or proc.stdout)[-2000:]
        except subprocess.TimeoutExpired:
            return False, "Execution timed out after 8 seconds"


def evaluate_task(backend: Any, category: str, task: dict, thinking: bool) -> dict:
    started = time.perf_counter()
    result = backend.chat(task["prompt"], thinking=thinking, max_tokens=768 if category in {"coding", "agentic"} else 384)
    elapsed = time.perf_counter() - started
    answer = result.text
    checks = task.get("checks", [])
    normalized = answer.lower()
    if task.get("all_checks", True):
        passed = all(c.lower() in normalized for c in checks)
    else:
        passed = any(c.lower() in normalized for c in checks)
    details = "text checks passed" if passed else "one or more expected markers missing"
    if "code_tests" in task:
        passed, details = run_code_tests(extract_code(answer), task["code_tests"])
    return {
        "id": task["id"], "category": category, "passed": bool(passed),
        "seconds": round(elapsed, 3), "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens, "details": details,
        "prompt": task["prompt"], "response": answer,
    }


def run_agent_challenge(backend: Any) -> dict:
    """A bounded tool-use loop with a local calculator and explicit step cap."""
    prompt = '''You are an assistant with one tool: calculator(expression), which evaluates basic arithmetic.
Return exactly one JSON object: {"tool":"calculator","expression":"..."} when you need the tool, or {"final":"answer"} when done.
Solve this: (19 * 23) + (144 / 12).'''
    started = time.perf_counter()
    first = backend.chat(prompt, max_tokens=180).text
    match = re.search(r"\{.*?\}", first, re.S)
    record = {"id": "bounded_tool_loop", "category": "agentic", "response": first}
    if not match:
        record.update(passed=False, details="Model did not return a parseable JSON tool/final action", seconds=round(time.perf_counter()-started, 3))
        return record
    try:
        action = json.loads(match.group())
        if action.get("tool") != "calculator":
            raise ValueError("Expected calculator tool action")
        expression = action.get("expression", "")
        if not re.fullmatch(r"[0-9+*/(). \-]+", expression):
            raise ValueError("Calculator expression contained disallowed characters")
        value = eval(expression, {"__builtins__": {}}, {})
        followup = backend.chat(f"The calculator returned {value}. Give only the final numeric answer.", max_tokens=64).text
        passed = "449" in followup
        record.update(passed=passed, details=f"calculator={value}; final={followup}", response=first + "\nTOOL RESULT: " + str(value) + "\nFINAL: " + followup)
    except Exception as exc:
        record.update(passed=False, details=f"Tool execution rejected/failed: {exc}")
    record["seconds"] = round(time.perf_counter()-started, 3)
    return record


def self_improve_challenge(backend: Any) -> dict:
    """One bounded critique/revision cycle; scores against explicit requirements."""
    started = time.perf_counter()
    task = "Write a Python function normalize_name(name) that trims whitespace, collapses internal whitespace to one space, and title-cases words. Return only code."
    draft = backend.chat(task, max_tokens=220).text
    critique = backend.chat(
        "Review this code against the requirements: trim ends, collapse repeated whitespace, title-case words. Identify concrete defects only.\n" + draft,
        max_tokens=180,
    ).text
    revised = backend.chat(
        "Revise the code using this critique. Return only Python code.\nDRAFT:\n" + draft + "\nCRITIQUE:\n" + critique,
        max_tokens=260,
    ).text
    code = extract_code(revised)
    tests = [{"input": "assert normalize_name('  aNURAG   pRASAD ') == 'Anurag Prasad'\nassert normalize_name('one\\ttwo') == 'One Two'\nassert normalize_name('') == ''"}]
    passed, details = run_code_tests(code, tests)
    return {"id": "critique_revise_cycle", "category": "self_improve", "passed": passed,
            "seconds": round(time.perf_counter()-started, 3), "details": details,
            "draft": draft, "critique": critique, "response": revised}


def main() -> None:
    parser = argparse.ArgumentParser(description="SenseNova Loop-9B evaluation harness")
    parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    parser.add_argument("--model", default="nativ-community/SenseNova-Loop-9B-4bit")
    parser.add_argument("--categories", default="all", help="all or comma-separated: basic,reasoning,coding,agentic,longcontext,self_improve")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", default="reports/latest.json")
    args = parser.parse_args()

    selected = list(TASKS) + ["self_improve"] if args.categories == "all" else [x.strip() for x in args.categories.split(",")]
    unknown = set(selected) - (set(TASKS) | {"self_improve"})
    if unknown:
        parser.error(f"Unknown categories: {', '.join(sorted(unknown))}")

    backend = load_backend(args.backend, args.model, args.max_tokens)
    records = []
    for category in selected:
        print(f"\n### {category.upper()} ###", flush=True)
        if category == "self_improve":
            record = self_improve_challenge(backend)
            records.append(record)
            print(f"{'PASS' if record['passed'] else 'FAIL'} {record['id']} ({record['seconds']}s)", flush=True)
            continue
        for task in TASKS[category]:
            try:
                record = evaluate_task(backend, category, task, args.thinking)
            except Exception as exc:
                record = {"id": task["id"], "category": category, "passed": False, "details": repr(exc), "seconds": 0}
            records.append(record)
            print(f"{'PASS' if record['passed'] else 'FAIL'} {record['id']} ({record['seconds']}s): {record['details']}", flush=True)

    if "agentic" in selected:
        record = run_agent_challenge(backend)
        records.append(record)
        print(f"{'PASS' if record['passed'] else 'FAIL'} {record['id']}: {record['details']}", flush=True)

    total = len(records)
    passed = sum(bool(r.get("passed")) for r in records)
    report = {"model": args.model, "backend": args.backend, "thinking": args.thinking,
              "max_tokens": args.max_tokens, "python": platform.python_version(),
              "platform": platform.platform(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "summary": {"passed": passed, "total": total, "pass_rate": round(passed / total, 4) if total else 0},
              "results": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOverall: {passed}/{total} ({report['summary']['pass_rate']:.1%})")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
