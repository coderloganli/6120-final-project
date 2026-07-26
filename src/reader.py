"""Reader: answer generator. Fixed across all experiments.

Runs a local instruct model. Given the question and retrieved memories, returns
a short answer.
"""
import os
from typing import List

from src.schema import Memory, Reader

# Default model. Override with READER_MODEL for larger or smaller hardware.
MODEL = os.environ.get("READER_MODEL", "Qwen/Qwen2.5-3B-Instruct")
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # left-pad so generations align (decoder-only)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    def _messages(self, question: str, context: List[Memory]):
        ctx = _join_context(context, MAX_CONTEXT_TOKENS) or "no context"
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ]

    def answer_batch(self, questions: List[str], contexts: List[List[Memory]]) -> List[str]:
        convos = [self._messages(q, c) for q, c in zip(questions, contexts)]
        inputs = self.tokenizer.apply_chat_template(
            convos, add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def answer(self, question: str, context: List[Memory]) -> str:
        return self.answer_batch([question], [context])[0]
