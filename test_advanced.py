# test_advanced.py
import torch
import time
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "sensenova/SenseNova-Loop-9B" # Use the final release for full capabilities

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()
print("Model loaded successfully.\n")

# --- Helper function for text-only chat ---
def run_chat(prompt_text, enable_thinking=False, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.time() - start_time
    response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    input_tokens = inputs.input_ids.shape[1]
    output_tokens = outputs.shape[1] - input_tokens
    print(f"  [Input tokens: {input_tokens} | Output tokens: {output_tokens} | Time: {elapsed:.1f}s]")
    return response.strip()

# ============================================================
# TEST 1: Reasoning Mode
# ============================================================
print("=" * 60)
print("TEST 1: Reasoning Mode (enable_thinking)")
print("=" * 60)
prompt = "A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step."

print("\n--- Without Thinking ---")
response_direct = run_chat(prompt, enable_thinking=False, max_new_tokens=128)
print(response_direct)

print("\n--- With Thinking (may take several minutes on CPU) ---")
response_think = run_chat(prompt, enable_thinking=True, max_new_tokens=512)
print(response_think)

# ============================================================
# TEST 2: Multimodal Image Understanding
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Multimodal Image Understanding")
print("=" * 60)

# Create a simple synthetic image: red square with white circle
img = Image.new("RGB", (256, 256), color="red")
draw = ImageDraw.Draw(img)
draw.ellipse([64, 64, 192, 192], fill="white")
image_path = "/tmp/test_image.png"
img.save(image_path)
print(f"Created synthetic image at {image_path}")

messages_mm = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image in one sentence. What are the dominant colors and shapes?"},
        ],
    }
]
prompt_mm = tokenizer.apply_chat_template(
    messages_mm,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs_mm = tokenizer(prompt_mm, return_tensors="pt").to(model.device)

start_time = time.time()
with torch.inference_mode():
    outputs_mm = model.generate(
        **inputs_mm,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
elapsed_mm = time.time() - start_time
response_mm = tokenizer.decode(outputs_mm[0, inputs_mm.input_ids.shape[1]:], skip_special_tokens=True)
print(f"[Time: {elapsed_mm:.1f}s]")
print(response_mm.strip())

# ============================================================
# TEST 3: Long-Context Needle-in-a-Haystack
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Long-Context Needle-in-a-Haystack (~8K tokens)")
print("=" * 60)

filler = "The quick brown fox jumps over the lazy dog. This is a completely unrelated sentence. "
needle = "The secret access code for the vault is BLUE-FALCON-77. "
# Build a haystack of roughly 8000 tokens
haystack = filler * 300  # Approx 300 repetitions
insert_point = len(haystack) // 2
long_context = haystack[:insert_point] + needle + haystack[insert_point:]

question = "\n\nBased on the text above, what is the secret access code for the vault? Reply with the code only."
prompt_long = long_context + question
response_long = run_chat(prompt_long, enable_thinking=False, max_new_tokens=32)
print(response_long)

print("\n" + "=" * 60)
print("Advanced tests complete.")
print("=" * 60)
