from dataclasses import dataclass
import sys
sys.path.append("../")
from data_loader.tasks import Instance
from models.models import Model
from typing import List

@dataclass
class LLMGen(Instance):
    """
        Represents a single instance of a task for generating text using an LLM.
    """
    models : Model
    llm_response : List[str]

