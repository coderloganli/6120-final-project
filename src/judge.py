"""Judge: QA scorer. Fixed across all experiments.

Runs a local instruct model. Returns 1.0 if the predicted answer matches the
gold answer, else 0.0. Local, so the project runs without API access.
"""
import os

from src.schema import Judge

# Default model. Override with JUDGE_MODEL.
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

_SYSTEM = (
    "You grade a question-answering system. Decide whether the predicted answer "
    "means the same as the gold answer. Reply with exactly one word: yes or no."
)
_USER = "Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"


class LocalLLMJudge(Judge):
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

    def score(self, question: str, pred: str, gold: str) -> float:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(question=question, gold=gold, pred=pred)},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        verdict = self.tokenizer.decode(gen, skip_special_tokens=True).strip().lower()
        return 1.0 if verdict.startswith("y") else 0.0
