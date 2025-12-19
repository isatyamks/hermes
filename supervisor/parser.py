import json


def parse_llm_response(text: str):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM output is not valid JSON:\n" + text
        ) from e

    required = {"diagnosis", "suggested_actions", "confidence"}
    if not required.issubset(data):
        raise ValueError("Missing required fields in LLM output")

    return data
