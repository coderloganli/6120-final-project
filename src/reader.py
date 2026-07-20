"""Reader: answer generator. Fixed across all experiments.

Runs a local instruct model. Given the question and retrieved memories, returns
a short answer.
"""
import os
from typing import List

from src.schema import Memory, Reader

# Default model. Override with READER_MODEL for smaller hardware.
MODEL = os.environ.get("READER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_CONTEXT_TOKENS = 1500   # context token cap, keeps extractors comparable
MAX_NEW_TOKENS = 64

_SYSTEM = (
    "Answer the question using only the context. "
    "If the context does not contain the answer, say you don't know. "
    "Answer as briefly as possible."
)


def _join_context(memories: List[Memory], max_tokens: int) -> str:
    out, used = [], 0
    for m in memories:
        n = len(m.text.split())
        if used + n > max_tokens:
            break
        out.append(m.text)
        used += n
    return "\n".join(out)


class LocalLLMReader(Reader):
    def __init__(self, model_name: str = MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    def answer(self, question: str, context: List[Memory]) -> str:
        ctx = _join_context(context, MAX_CONTEXT_TOKENS) or "no context"
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
