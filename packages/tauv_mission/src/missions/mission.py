from abc import ABC, abstractmethod
from typing import Dict, Optional
from tasks.task import Task, TaskStatus, TaskResult


class Mission(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def entrypoint(self) -> Optional[Task]:
        pass

    @abstractmethod
    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        pass
