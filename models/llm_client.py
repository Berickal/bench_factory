from abc import ABC, abstractmethod
from models.models import Model, Platform

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from transformers.utils import logging as transformers_logging

from openai import OpenAI
from ollama import Client

import os
import logging

transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

class LlmClient(ABC):
    _model: Model

    def get_model_name(self) -> str:
        return self._model.name

    @abstractmethod
    def send_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        pass


class OpenAIClient(LlmClient):

    def __init__(self, model: Model):
        self._model = model
        self._client = OpenAI(
            base_url=model.base_url,
            api_key=str(os.getenv("OPENAI_API_KEY")),
        )

    def send_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        completion = self._client.chat.completions.create(
            model=self._model.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        print("Get result")
        return completion.choices[0].message.content
    

class OllamaClient(LlmClient):
    def __init__(self, model: Model):
        self._model = model
        self._client = Client(host=model.base_url)

    def send_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        completion = self._client.chat(
            model=self._model.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.message.content
   
class HuggingFaceClient(LlmClient):
    def __init__(self, model: Model):
        self._model = model
        self._tokenizer = None
        self._model_instance = None
        self._pipeline = None
        self._load_model()

    def _get_device_map(self):
        """Determine the appropriate device mapping."""
        if self._model.device == "auto":
            if torch.cuda.is_available():
                return "auto"
            else:
                return "cpu"
        return self._model.device

    def _get_torch_dtype(self):
        """Get the appropriate torch dtype."""
        if self._model.torch_dtype == "auto":
            if torch.cuda.is_available():
                return torch.float16
            else:
                return torch.float32
        elif self._model.torch_dtype == "float16":
            return torch.float16
        elif self._model.torch_dtype == "bfloat16":
            return torch.bfloat16
        elif self._model.torch_dtype == "float32":
            return torch.float32
        else:
            return torch.float16

    def _setup_quantization(self):
        """Setup quantization config if needed (optional for memory efficiency)."""
        if torch.cuda.is_available() and self._model.torch_dtype in ["float16", "bfloat16"]:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None

    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            print(f"Loading HuggingFace model: {self._model.model_path}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model.model_path,
                trust_remote_code=True,
                token=os.getenv(self._model.api_key_variable_name) if self._model.api_key_variable_name else None
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            quantization_config = self._setup_quantization()

            self._model_instance = AutoModelForCausalLM.from_pretrained(
                self._model.model_path,
                torch_dtype=self._get_torch_dtype(),
                device_map=self._get_device_map(),
                trust_remote_code=True,
                quantization_config=quantization_config,
                token=os.getenv(self._model.api_key_variable_name) if self._model.api_key_variable_name else None
            )

            # Create text generation pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self._model_instance,
                tokenizer=self._tokenizer,
                torch_dtype=self._get_torch_dtype(),
                device_map=self._get_device_map(),
            )

            print(f"Successfully loaded model: {self._model.name}")

        except Exception as e:
            print(f"Error loading HuggingFace model {self._model.model_path}: {e}")
            raise

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format the prompt based on the model's expected format."""
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                return self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        if system_prompt.strip():
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            return f"{user_prompt}"

    def send_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        """Send a prompt to the HuggingFace model and get response."""
        try:
            # Format the prompt
            formatted_prompt = self._format_prompt(system_prompt, user_prompt)
            
            # Generate response using pipeline
            outputs = self._pipeline(
                formatted_prompt,
                max_new_tokens=self._model.max_new_tokens,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                return_full_text=False,  # Only return the generated part
            )

            # Extract the generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                return generated_text.strip()
            else:
                print("No output generated")
                return None

        except Exception as e:
            print(f"Error generating response with HuggingFace model: {e}")
            return None

    def __del__(self):
        """Cleanup resources when the client is destroyed."""
        try:
            if hasattr(self, '_model_instance') and self._model_instance is not None:
                del self._model_instance
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                del self._tokenizer
            if hasattr(self, '_pipeline') and self._pipeline is not None:
                del self._pipeline
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
def get_client_for_model(model: Model) -> LlmClient:
    match model.platform:
        case Platform.OLLAMA:
            return OllamaClient(model)
        case Platform.OPENAI:
            return OpenAIClient(model)
        case Platform.HUGGINGFACE:
            return HuggingFaceClient(model)
        case _:
            raise Exception(
                f"Instance of `LlmClient` not implemented for platform: {model.platform}"
            )
