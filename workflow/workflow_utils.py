"""Workflow utilities for Retrieval-Augmented Generation (RAG) systems."""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type

import graphviz

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(self, name: str, func: Callable[..., Any], step_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.func = func
        self.step_config = step_config or {}

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


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


def draw_all_possible_flows(cls: Type, filename: str = "workflow_flows", format: str = "png") -> None:
    """
    Visualizes all possible flows within a workflow class based on defined steps.

    Args:
        cls (Type): The workflow class to visualize.
        filename (str, optional): The filename for the output graph. Defaults to "workflow_flows".
        format (str, optional): The format of the output graph. Defaults to "png".

    """
    steps = get_steps_from_class(cls)
    dot = graphviz.Digraph(comment="Workflow Flows")

    for step_name, step in steps.items():
        dot.node(step_name, step_name)

    # For simplicity, connect steps in the order they are defined
    step_names = list(steps.keys())
    for i in range(len(step_names) - 1):
        dot.edge(step_names[i], step_names[i + 1])

    dot.render(filename, format=format, cleanup=True)
    logger.info(f"All possible workflow flows have been saved to {filename}.{format}")


def draw_most_recent_execution(workflow_instance: Any, filename: str = "recent_workflow_execution", format: str = "png") -> None:
    """
    Visualizes the most recent execution path of the workflow.

    Args:
        workflow_instance (Any): The instance of the workflow that was executed.
        filename (str, optional): The filename for the output graph. Defaults to "recent_workflow_execution".
        format (str, optional): The format of the output graph. Defaults to "png".

    """
    if not hasattr(workflow_instance, "execution_history"):
        logger.warning("No execution history found in the workflow instance.")
        return

    history = workflow_instance.execution_history
    dot = graphviz.Digraph(comment="Recent Workflow Execution")

    for step in history:
        dot.node(step, step)

    for i in range(len(history) - 1):
        dot.edge(history[i], history[i + 1])

    dot.render(filename, format=format, cleanup=True)
    logger.info(f"Most recent workflow execution has been saved to {filename}.{format}")