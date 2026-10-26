"""Verifier for QA: compare prediction to ground truth. Uses VERIFIER_API_KEY and VERIFIER_END_POINT from env."""

import re
import os
import time
import openai

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
    "for question-answer pairs. Compare the predicted answer with the correct answer and determine "
    "if they match meaningfully. Consider synonyms or paraphrases as valid matches."
)

QUERY_PROMPT = """
1. **Question**: {question}
2. **Ground Truth Answer**: {ground_truth}
3. **Model Predicted Answer**: {prediction}

Evaluate the model's prediction against the ground truth. Output an integer score: 1 for correct, 0 for incorrect.
Respond using exactly: Score: 1 or Score: 0
Explanation: <your explanation>"""


class GPT4VisionClient:
    """Client for verifier API. Credentials from env: VERIFIER_API_KEY, VERIFIER_END_POINT, VERIFIER_MODEL_NAME."""

    def __init__(self, endpoint=None, api_key=None):
        self.api_key = os.environ.get("VERIFIER_API_KEY")
        self.end_point = os.environ.get("VERIFIER_END_POINT")
        self.model_name = os.environ.get("VERIFIER_MODEL_NAME", "gpt-4o")
        self.client = openai.OpenAI(base_url=self.end_point, api_key=self.api_key)

    def query(self, prompt: str, system_prompt: str = None, max_retries=5, initial_delay=3):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        for attempt in range(max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.2, max_tokens=8192, timeout=120
                )
                text = r.choices[0].message.content
                if "score:" not in text.lower():
                    raise ValueError("No score in response")
                part = text.lower().split("score:")[-1].strip().split("\n")[0].strip().split()[0]
                if "1" not in part and "0" not in part:
                    raise ValueError("No 0/1 in response")
                return part, text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(min(initial_delay * (attempt + 1), 10))
                else:
                    return "", str(e)
        return "", ""


def compute_score(prompt: str, predict_str_list: list, ground_truth, extra_info: dict = None):
    """Return (accuracy_score, analysis). Uses verifier API if VERIFIER_API_KEY and VERIFIER_END_POINT are set."""
    extra = extra_info or {}
    if not os.environ.get("VERIFIER_API_KEY") or not os.environ.get("VERIFIER_END_POINT"):
        return 0.0, "Verifier not configured (set VERIFIER_API_KEY and VERIFIER_END_POINT to enable)."
    full = " ".join(predict_str_list)
    if extra.get("gpt_extract_answer") and extra.get("extract_answer_tags") == "strict":
        matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", full, re.DOTALL)
        full = matches[-1].strip() if matches else (predict_str_list[-1].strip() if predict_str_list else "")
    query = QUERY_PROMPT.format(question=prompt, ground_truth=ground_truth, prediction=full)
    client = GPT4VisionClient()
    try:
        score_part, response_text = client.query(query, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        return 0.0, f"Evaluation error: {e}"
    acc = 1.0 if score_part and "1" in score_part else 0.0
    analysis = ""
    if response_text and "explanation:" in response_text.lower():
        analysis = response_text.lower().split("explanation:")[-1].strip()
    return extra.get("acc_reward_weight", 1.0) * acc, analysis or "No explanation"
