"""Judge: QA scorer. Fixed across all experiments.

Runs a local instruct model. Returns 1.0 if the predicted answer matches the
gold answer, else 0.0. Local, so the project runs without API access.
"""
import os

from src.schema import Judge

# A different family from the reader avoids self-preference. Override with JUDGE_MODEL. 
MODEL = os.environ.get("JUDGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# One user turn, no system role, so any chat template works.
_PROMPT = (
    "You grade a question-answering system. Decide whether the predicted answer "
    "means the same as the gold answer. Reply with exactly one word: yes or no.\n\n"
    "Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"
)


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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # left-pad so generations align (decoder-only)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    def score_batch(self, triples) -> list:
        convos = [
            [{"role": "user", "content": _PROMPT.format(question=q, gold=g, pred=p)}]
            for q, p, g in triples
        ]
        inputs = self.tokenizer.apply_chat_template(
            convos, add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        verdicts = [self.tokenizer.decode(g, skip_special_tokens=True).strip().lower() for g in gen]
        return [1.0 if v.startswith("y") else 0.0 for v in verdicts]

    def score(self, question: str, pred: str, gold: str) -> float:
        return self.score_batch([(question, pred, gold)])[0]
