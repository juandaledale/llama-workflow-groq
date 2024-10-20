# workflow_utils.py

"""Workflow utilities for RAG systems using llama_index."""

import inspect
from typing import Any, Callable, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(self, name: str, func: Callable[..., Any], step_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.func = func
        self.step_config = step_config or {}

    async def __call__(self, *args, **kwargs) -> Any:
        return await self.func(*args, **kwargs)


def get_steps_from_class(cls: Type) -> Dict[str, WorkflowStep]:
    """
    Retrieves workflow steps defined as methods of a class.

    Args:
        cls (Type): The class to inspect for workflow steps.

    Returns:
        Dict[str, WorkflowStep]: A dictionary mapping step names to WorkflowStep instances.
    """
    steps = {}
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        step_config = getattr(method, "__step_config", None)
        if step_config is not None:
            steps[name] = WorkflowStep(name, method, step_config)
    logger.debug(f"Extracted steps from class '{cls.__name__}': {list(steps.keys())}")
    return steps


def get_steps_from_instance(instance: Any) -> Dict[str, WorkflowStep]:
    """
    Retrieves workflow steps defined as bound methods of an instance.

    Args:
        instance (Any): The instance to inspect for workflow steps.

    Returns:
        Dict[str, WorkflowStep]: A dictionary mapping step names to WorkflowStep instances.
    """
    steps = {}
    for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
        step_config = getattr(method, "__step_config", None)
        if step_config is not None:
            steps[name] = WorkflowStep(name, method, step_config)
    logger.debug(f"Extracted steps from instance '{type(instance).__name__}': {list(steps.keys())}")
    return steps


def step_config(**config):
    """
    Decorator to add configuration metadata to workflow step methods.

    Usage:
        @step_config(description="Ingest data from directory.")
        async def ingest(self, dirname: str):
            ...

    Args:
        **config: Arbitrary keyword arguments representing configuration.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        setattr(func, "__step_config", config)
        return func
    return decorator