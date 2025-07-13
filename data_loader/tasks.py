from dataclasses import dataclass
from typing import Optional, Dict, List

from abc import ABC, abstractmethod

@dataclass
class Instance:
    """
        Represents a single instance of a task for benchmarking an LLM.
    """
    input : str
    ref_output : str
    metadata : Optional[Dict[str, str]]


@dataclass
class Benchmark(ABC):
    """
        Represents a benchmark for evaluating an LLM task.
    """
    name : str
    publisher : str
    url : str
    data : List[Instance] = None
    type : str = "CODE_SYNTHESIS"

    @abstractmethod
    def map_to_instance(data : any) -> Instance:
        """
            Maps raw data to an Instance.
        """
        pass

    @abstractmethod
    def load_benchmark() -> 'Benchmark':
        """
            Loads the benchmark data.
        """
        pass


@dataclass
class Task:
    """
        Represents a task that can be performed by an LLM.
    """
    name : str
    description : str
    system_prompt : str
    policy : Optional[str] = None
    benchmarks: List[Benchmark] = None
    