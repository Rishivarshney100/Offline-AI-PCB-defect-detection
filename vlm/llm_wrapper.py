from typing import Optional
import time


class LLMWrapper:
    def __init__(self, backend: str = "ollama", model_name: str = "phi-2",
                 base_url: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 256):
        self.backend = backend
        self.model_name = model_name
        self.base_url = base_url or "http://localhost:11434"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        if self.backend == "ollama":
            self._init_ollama()
        elif self.backend == "llama_cpp":
            self._init_llama_cpp()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _init_ollama(self):
        try:
            import ollama
            self.llm = ollama
            try:
                models = ollama.list()
                if self.model_name not in [m['name'] for m in models.get('models', [])]:
                    print(f"Warning: Model {self.model_name} not found. Install with: ollama pull {self.model_name}")
            except:
                pass
        except ImportError:
            raise ImportError("Ollama not installed. Install with: pip install ollama")
    
    def _init_llama_cpp(self):
        try:
            from llama_cpp import Llama
            raise NotImplementedError("llama.cpp requires model_path configuration")
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    def _init_transformers(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.llm = self.llm.to("cpu")
            print("Model loaded")
        except ImportError:
            raise ImportError("Transformers not installed. Install with: pip install transformers accelerate")
    
    def generate(self, prompt: str, **kwargs) -> str:
        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        start = time.time()
        
        if self.backend == "ollama":
            response = self._generate_ollama(prompt, temp, max_tok)
        elif self.backend == "llama_cpp":
            response = self._generate_llama_cpp(prompt, temp, max_tok)
        elif self.backend == "transformers":
            response = self._generate_transformers(prompt, temp, max_tok)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        if time.time() - start > 1.0:
            print(f"LLM generation took {time.time() - start:.2f}s")
        return response
    
    def _generate_ollama(self, prompt: str, temp: float, max_tok: int) -> str:
        try:
            return self.llm.generate(model=self.model_name, prompt=prompt,
                                   options={"temperature": temp, "num_predict": max_tok}).get("response", "")
        except:
            try:
                import requests
                r = requests.post(f"{self.base_url}/api/generate", json={
                    "model": self.model_name, "prompt": prompt,
                    "options": {"temperature": temp, "num_predict": max_tok}
                }, timeout=30)
                return r.json().get("response", "")
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}")
    
    def _generate_llama_cpp(self, prompt: str, temp: float, max_tok: int) -> str:
        raise NotImplementedError("llama.cpp backend not fully implemented")
    
    def _generate_transformers(self, prompt: str, temp: float, max_tok: int) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.llm, "device"):
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=max_tok,
                                      temperature=temp, do_sample=temp > 0,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text
    
    def is_available(self) -> bool:
        try:
            if self.backend == "ollama":
                import requests
                return requests.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
            return self.llm is not None if self.backend == "transformers" else False
        except:
            return False
