import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orbitdesk_agent.state import RetrievedDocument


class LocalGenerator:
    def __init__(self, model_name: str, revision: str, offline: bool) -> None:
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=offline,
            torch_dtype="auto",
        )
        self.model.eval()
        self.model_name = model_name
        self.resolved_revision = self.model.config._commit_hash or revision
        self.device = str(self.model.device)
        self.load_seconds = round(time.perf_counter() - started, 2)

    def answer(self, query: str, documents: list[RetrievedDocument]) -> tuple[str, float]:
        evidence = "\n\n".join(
            f"[{document['source_id']}] {document['passage']}" for document in documents
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an OrbitDesk support assistant. Use only the supplied evidence. "
                    "Do not invent troubleshooting steps, teams, servers, actions, or policies. "
                    "Do not mention customer service, IT support, or external systems unless the evidence does. "
                    "Do not repeat the question, add labels, or include source IDs in the answer. "
                    "Answer every part of the question. Include documented limitations and the documented next action. "
                    "Keep the answer under 100 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nEvidence:\n{evidence}"
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {
            name: value.to(self.model.device)
            for name, value in self.tokenizer(prompt, return_tensors="pt").items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer or "I could not produce a grounded answer from the available evidence.", round(
            time.perf_counter() - started,
            2,
        )
