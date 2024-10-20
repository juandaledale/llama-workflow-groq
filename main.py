"""Main script to execute the RAG Workflow."""

import asyncio
import nest_asyncio
import logging

from workflow.embedder import Embedder  # Assuming you have an Embedder implementation
from workflow.vector_store import VectorStore  # Assuming you have a VectorStore implementation
from workflow.workflow_utils import draw_all_possible_flows, draw_most_recent_execution
from workflow.rag_workflow import ConcreteRAGWorkflow

# Apply nest_asyncio to allow nested event loops (useful in some environments like Jupyter)
nest_asyncio.apply()

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag_main")


async def main():
    # Initialize components
    embedder = Embedder()  # Replace with your actual Embedder instance
    vector_store = VectorStore()  # Replace with your actual VectorStore instance

    # Instantiate the workflow
    workflow = ConcreteRAGWorkflow(embedder=embedder, vector_store=vector_store)

    # Retrieve and inspect the registered steps
    steps = workflow.steps
    print(f"Registered steps: {list(steps.keys())}")
    for step_name, step in steps.items():
        print(f"Step '{step_name}' config: {step.step_config}")

    # Visualize all possible flows
    draw_all_possible_flows(ConcreteRAGWorkflow, filename="multi_step_workflow", format="png")

    # Run the ingest step
    if hasattr(workflow, "ingest"):
        ingest_result = await workflow.run(dirname="Data")
    else:
        logger.error("Ingest step is not defined in the workflow.")
        return

    # Run the retrieve, rerank, and synthesize steps
    if hasattr(workflow, "retrieve") and hasattr(workflow, "rerank") and hasattr(workflow, "synthesize"):
        rag_result = await workflow.run(query="How much is the project?", index=ingest_result)
    else:
        logger.error("Retrieve, rerank, or synthesize steps are not defined in the workflow.")
        return

    # Print the result
    print(f"\nSynthesized Answer: {rag_result}")

    # Visualize the most recent execution
    draw_most_recent_execution(workflow, filename="rag_flow_recent", format="png")


if __name__ == "__main__":
    asyncio.run(main())