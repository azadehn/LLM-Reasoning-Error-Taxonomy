"""
OpenAI model runner for collecting reasoning outputs.
This script focuses on reproducible prompt execution and structured logging.
"""

import json
import os
from typing import Dict

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_prompt(prompt: str, model: str = "gpt-4o-mini") -> Dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Solve the problem step by step and explain your reasoning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return {
        "model": model,
        "prompt": prompt,
        "response": response.choices[0].message.content
    }


if __name__ == "__main__":
    example_prompt = "If 2 + 3 * 4 = ?, explain your reasoning."
    output = run_prompt(example_prompt)
    print(json.dumps(output, indent=2))
