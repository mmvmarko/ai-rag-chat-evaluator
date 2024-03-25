import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from azure.ai.generative.evaluate import evaluate

from . import service_setup
from .evaluate_metrics import metrics_by_name

logger = logging.getLogger("scripts")

def send_question_to_target(question: str, truth: str, target_url: str, parameters: dict = {}, raise_error=False):
    headers = {"Content-Type": "application/json"}
    body = {
        "messages": [{"content": question, "role": "user"}],
        "stream": False,
        "context": parameters,
    }
    try:
        r = requests.post(target_url, headers=headers, json=body)
        r.encoding = "utf-8"

        latency = r.elapsed.total_seconds()
        
        # Check if response body is empty
        if not r.text.strip():
            raise ValueError("Received empty response body from target URL.")

        response_dict = r.json()

        try:
            answer = response_dict["choices"][0]["message"]["content"]
            data_points = response_dict["choices"][0]["context"]["data_points"]["text"]
            context = "\n\n".join(data_points)
        except Exception:
            raise ValueError(
                "Response does not adhere to the expected schema. "
                "Either adjust the app response or adjust send_question_to_target() in evaluate.py "
                f"to match the actual schema.\nResponse: {response_dict}"
            )

        response_obj = {"question": question, "truth": truth, "answer": answer, "context": context, "latency": latency}
        return response_obj
    except Exception as e:
        if raise_error:
            raise e
        return {
            "question": question,
            "truth": truth,
            "answer": str(e),
            "context": str(e),
            "latency": -1,
        }

# Remaining parts of evaluate.py stay the same, including the definitions of truncate_for_log, load_jsonl, run_evaluation, process_config, and run_evaluate_from_config functions.
