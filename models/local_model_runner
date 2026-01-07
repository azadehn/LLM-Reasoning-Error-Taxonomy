"""
Local LLM runner using HuggingFace transformers.
Supports offline evaluation and reproducibility.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def run_prompt(prompt: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer, model = load_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "Explain why the conclusion does or does not follow."
    print(run_prompt(prompt))
