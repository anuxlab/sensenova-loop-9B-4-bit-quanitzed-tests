# test_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the official model ID. The Preview version is also available.
model_id = "sensenova/SenseNova-Loop-9B-Preview" # Or the final "sensenova/SenseNova-Loop-9B"

# Load tokenizer and model
# device_map="auto" will default to CPU if no GPU is available.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, # Use bfloat16 for stability on CPU
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

# --- Text-only test ---
messages = [{"role": "user", "content": "What is 13 times 7? Reply with the number only."}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False, # Set to True to enable the model's thinking mode
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Running inference...")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64, # Keep it small to save time on CPU
        do_sample=False, # Greedy decoding for reproducibility
        use_cache=True,
    )

# Decode the response, stripping the input prompt tokens
response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- Model Output ---")
print(response)
print("--------------------")
