from dataclasses import dataclass
from enum import Enum


class Platform(str, Enum):
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    HUGGINGFACE = "HuggingFace"


@dataclass
class Model:
    name: str
    developer: str
    base_url: str
    api_key_variable_name : str
    platform : Platform

    model_path: str = None  # For local models or HF model identifiers
    device: str = "auto"  # cuda, cpu, or auto
    torch_dtype: str = "auto"  # float16, bfloat16, float32, or auto
    max_new_tokens: int = 2048

    def __iter__(self):
        return iter(
            (
                self.name,
                self.developer,
                self.base_url,
                self.api_key_variable_name,
                self.platform,
            )
        )
    
MODEL_NAME = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-3.5-turbo-0125"
]

OPENAI_MODEL = [
    Model(
        name=model_name,
        developer="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_variable_name="OPENAI_API_KEY",
        platform=Platform.OPENAI,
    ) for model_name in MODEL_NAME
]

OLLAMA_MODEL = [
    "deepseek-r1:14b",
    "deepseek-coder-v2:latest"
    #Complete with other Ollama models as needed
]

OLLAMA_MODEL = [
    Model(
        name=model_name,
        developer="Ollama",
        base_url="http://trux-hikari.uni.lux:11434/", ## Update with your Ollama server URL
        api_key_variable_name="OLLAMA_API_KEY",
        platform=Platform.OLLAMA,
    ) for model_name in OLLAMA_MODEL
]


HUGGINGFACE_MODEL = [
    Model(
        name="deepseek-coder-1.3b-instruct",
        developer="DeepSeek",
        base_url="",
        api_key_variable_name="HUGGINGFACE_API_KEY",
        platform=Platform.HUGGINGFACE,
        model_path="deepseek-ai/deepseek-coder-1.3b-instruct",
        device="auto",
        torch_dtype="bfloat16",
        max_new_tokens=2048,
    ),
    Model(
        name="codememo_epoch_1",
        developer="DeepSeek",
        base_url="",
        api_key_variable_name="HUGGINGFACE_API_KEY",
        platform=Platform.HUGGINGFACE,
        model_path="./../../models/checkpoint-500",
        device="auto",
        torch_dtype="bfloat16",
        max_new_tokens=2048,
    ),
    Model(
        name="codememo_epoch_100",
        developer="DeepSeek",
        base_url="",
        api_key_variable_name="HUGGINGFACE_API_KEY",
        platform=Platform.HUGGINGFACE,
        model_path="./../../models/checkpoint-27400",
        device="auto",
        torch_dtype="bfloat16",
        max_new_tokens=2048,
    ),
]

_MODELS = OPENAI_MODEL + OLLAMA_MODEL + HUGGINGFACE_MODEL
MODEL_NAMES_TO_MODELS = {model.name: model for model in _MODELS}