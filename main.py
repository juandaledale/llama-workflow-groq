# main.py

"""Main script to execute the RAG Workflow."""

import asyncio
import nest_asyncio
import logging
import sys
from typing import Any, Dict,Optional

from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution
from workflow.rag_workflow import ConcreteRAGWorkflow

# Apply nest_asyncio to allow nested event loops (useful in some environments like Jupyter)
nest_asyncio.apply()

# Configure Logging
def configure_logger(name: str = "rag_main") -> logging.Logger:
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger

# Placeholder implementations of Embedder and VectorStore
class Embedder:
    """Placeholder Embedder class."""
    async def get_text_embedding_batch(self, texts: list, show_progress: bool = False) -> list:
        await asyncio.sleep(0.1)  # Simulate async operation
        return [[0.1] * 768 for _ in texts]

    async def aget_text_embedding_batch(self, texts: list, show_progress: bool = False) -> list:
        return await self.get_text_embedding_batch(texts, show_progress)

class VectorStore:
    """Placeholder VectorStore class."""
    def __init__(self):
        self.embeddings: Dict[str, list] = {}

    async def add_embeddings(self, embeddings: Dict[str, list]) -> None:
        await asyncio.sleep(0.1)  # Simulate async operation
        self.embeddings.update(embeddings)
        logging.getLogger("vector_store").debug(f"Added embeddings: {embeddings}")

    async def query(self, query_embedding: list, top_k: int = 5) -> list:
        await asyncio.sleep(0.1)  # Simulate async operation
        return [f"doc{i}" for i in range(1, top_k + 1)]

async def main():
    """Main entry point for executing the RAG workflow."""
    logger = configure_logger()

    embedder = Embedder()  # Replace with your actual Embedder implementation
    vector_store = VectorStore()  # Replace with your actual VectorStore implementation

    workflow = ConcreteRAGWorkflow(embedder=embedder, vector_store=vector_store)

    steps = workflow.steps
    logger.info(f"Registered steps: {list(steps.keys())}")
    for step_name, step in steps.items():
        logger.info(f"Step '{step_name}' config: {step.step_config}")

    # Visualize workflow flows
    try:
        draw_all_possible_flows(workflow, filename="multi_step_workflow.html")
        logger.info("All possible workflow flows have been visualized in 'multi_step_workflow.html'.")
    except Exception as e:
        logger.error(f"Failed to visualize workflow flows: {e}")
    query = "resume the text"
    # Run the ingest step
    try:
        ingest_result = await workflow.run(dirname="Data", query=query, index=None)
    except ValueError as e:
        logger.error(f"Ingest step encountered an issue: {e}")
        return  # Early termination if ingest fails

    # IMPORTANT: Make sure to provide a valid query


    # Run the retrieve, rerank, and synthesize steps if ingest was successful
    try:
        rag_result = await workflow.run(dirname=None, query=query, index=ingest_result)
        logger.info(f"Synthesized Answer: {rag_result}")
    except ValueError as e:
        logger.error(f"RAG steps encountered an issue: {e}")

    # Visualize the most recent execution
    try:
        draw_most_recent_execution(workflow, filename="rag_flow_recent.html")
        logger.info("Most recent workflow execution has been visualized in 'rag_flow_recent.html'.")
    except Exception as e:
        logger.error(f"Failed to visualize recent workflow execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())