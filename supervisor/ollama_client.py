import subprocess


class OllamaClient:
    def __init__(self, model="llama3.2:3b"):
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        process = subprocess.Popen(
            ["ollama", "run", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )

        input_text = (
            system_prompt
            + "\n\n"
            + user_prompt
            + "\n\n"
            + "Respond ONLY with valid JSON."
        )

        stdout, stderr = process.communicate(input_text)

        if stderr and stderr.strip():
            print("Ollama stderr:", stderr)

        return stdout.strip()
