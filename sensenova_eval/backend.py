from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MLXBackend:
    """Native Apple-Silicon backend for the community 4-bit MLX-VLM export."""

    def __init__(self, model_id: str, max_tokens: int = 512):
        if os.uname().sysname != "Darwin" or os.uname().machine not in {"arm64", "aarch64"}:
            raise RuntimeError("The MLX backend requires macOS on Apple Silicon. Use --backend transformers elsewhere.")
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise RuntimeError("Install the MLX backend with: pip install -U mlx mlx-vlm") from exc
        self._generate = generate
        self._template = apply_chat_template
        print(f"Loading MLX model: {model_id}", flush=True)
        self.model, self.processor = load(model_id)
        self.model_id = model_id
        self.max_tokens = max_tokens

    def chat(self, prompt: str, *, thinking: bool = False, max_tokens: int | None = None) -> Generation:
        formatted = self._template(self.processor, self.model.config, prompt, enable_thinking=thinking)
        result = self._generate(
            self.model, self.processor, formatted,
            max_tokens=max_tokens or self.max_tokens,
            verbose=False,
        )
        return Generation(text=str(result.text).strip())

    def vision(self, image_path: str, prompt: str, *, max_tokens: int | None = None) -> Generation:
        formatted = self._template(
            self.processor, self.model.config, prompt,
            num_images=1, enable_thinking=False,
        )
        result = self._generate(
            self.model, self.processor, formatted, image=[image_path],
            max_tokens=max_tokens or self.max_tokens, verbose=False,
        )
        return Generation(text=str(result.text).strip())


class TransformersBackend:
    """Reference BF16/full-precision loader; it does not imply 4-bit quantization."""

    def __init__(self, model_id: str, max_tokens: int = 512):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install optional dependencies: pip install torch transformers accelerate safetensors") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto", low_cpu_mem_usage=True,
        ).eval()
        self.max_tokens = max_tokens

    def chat(self, prompt: str, *, thinking: bool = False, max_tokens: int | None = None) -> Generation:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens or self.max_tokens,
                do_sample=False, use_cache=True,
            )
        n_in = int(inputs.input_ids.shape[1])
        n_out = int(output.shape[1] - n_in)
        return Generation(self.tokenizer.decode(output[0, n_in:], skip_special_tokens=True).strip(), n_in, n_out)


def load_backend(name: str, model_id: str, max_tokens: int = 512) -> Any:
    if name == "mlx":
        return MLXBackend(model_id, max_tokens)
    if name == "transformers":
        return TransformersBackend(model_id, max_tokens)
    raise ValueError(f"Unknown backend: {name}")
