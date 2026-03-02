from langchain_ollama import Ollama

class LLMService:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "phi-2"):
        self.llm = Ollama(base_url=ollama_url, model=model)

    def chat(self, prompt: str):
        return self.llm.invoke(prompt)

