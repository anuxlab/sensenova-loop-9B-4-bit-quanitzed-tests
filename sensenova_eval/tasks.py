"""Small, deterministic starter tasks. Extend with licensed benchmark datasets."""
TASKS = {
    "basic": [
        {"id": "arith_01", "prompt": "What is 13 times 7? Reply with the number only.", "checks": ["91"]},
        {"id": "instruction_01", "prompt": "Reply with exactly these three comma-separated colors in lowercase: red, blue, green. Nothing else.", "checks": ["red", "blue", "green"]},
        {"id": "logic_01", "prompt": "A farmer has 17 sheep. All but 9 run away. How many remain? Reply with the number only.", "checks": ["9"]},
    ],
    "reasoning": [
        {"id": "logic_02", "prompt": "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Give the answer and a brief explanation.", "checks": ["0.05"]},
        {"id": "planning_01", "prompt": "Plan a safe three-step approach to debug a Python program that crashes only on empty input. Include a test for the empty case.", "checks": ["empty", "test"]},
    ],
    "coding": [
        {"id": "code_01", "prompt": "Write only Python code defining def is_prime(n): that returns True for prime integers and False otherwise. Handle n < 2.", "code_tests": [{"input": "assert is_prime(2)\nassert is_prime(17)\nassert not is_prime(1)\nassert not is_prime(0)\nassert not is_prime(21)"}]},
        {"id": "code_02", "prompt": "Write only Python code defining def flatten(items): that recursively flattens nested lists but leaves non-list values unchanged.", "code_tests": [{"input": "assert flatten([1,[2,[3]],4]) == [1,2,3,4]\nassert flatten([]) == []\nassert flatten([1,'x',[True]]) == [1,'x',True]"}]},
    ],
    "agentic": [
        {"id": "agent_01", "prompt": "You have tools named calculator(expression) and final(answer). Solve: (19 * 23) + (144 / 12). Return a JSON object with a tool_calls array; call calculator with the expression, then provide a final answer.", "checks": ["tool_calls", "calculator"]},
        {"id": "agent_02", "prompt": "A task has three stages: inspect input, validate it, then summarize it. The input is a CSV with missing values. Give an ordered execution plan and state what you would do if validation fails.", "checks": ["validate", "missing"]},
    ],
    "longcontext": [
        {"id": "needle_01", "prompt": "Reference notes: " + ("The project uses Python and pytest. Unrelated note: the sky is blue. " * 90) + "The private benchmark label is LOOP-ORCHID-824. " + ("More unrelated project notes. " * 90) + "\nWhat is the private benchmark label? Reply with the label only.", "checks": ["LOOP-ORCHID-824"]},
    ],
}
