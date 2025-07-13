from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Dict

@dataclass
class Metric(ABC):
    """
        Represent a metric used to evaluate the performance of a model given a DS task
    """
    name : str
    metric_range : any

    @abstractmethod
    def check_metric(input : str, metadata : Dict) -> float:
        pass

    @abstractmethod
    def evaluate(input : str, metadata : Dict) -> float:
        pass

@dataclass
class PassTest(Metric):
    pass

