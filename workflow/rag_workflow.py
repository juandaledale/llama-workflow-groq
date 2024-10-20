# rag_workflow.py

import os  # For directory operations
import asyncio
import logging
from typing import Any, Dict, List, Optional

from workflow.workflow_utils import WorkflowStep, step_config, get_steps_from_instance

logger = logging.getLogger("rag_workflow")
logging.basicConfig(level=logging.DEBUG)


class RAGWorkflow:
    """Retrieval-Augmented Generation Workflow."""

    def __init__(self):
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_history: List[str] = []
        self.accepted_events = {"start", "stop", "ingest", "retrieve", "rerank", "synthesize"}
        self._contexts = {}  # Initialize contexts if needed
        self.documents: List[str] = []  # To store ingested document contents
        self.register_steps()

    def register_steps(self):
        """Registers workflow steps from the instance."""
        self.steps = {name: step for name, step in get_steps_from_instance(self).items()}
        logger.debug(f"Registered steps: {list(self.steps.keys())}")

    async def run_step(self, step_name: str, **kwargs) -> Any:
        """Executes a single workflow step.

        Args:
            step_name (str): The name of the step to execute.
            **kwargs: Arbitrary keyword arguments for the step.

        Returns:
            Any: The result of the step execution.
        """
        if step_name not in self.steps:
            raise ValueError(f"Step {step_name} is not registered in the workflow.")

        step = self.steps[step_name]
        self.execution_history.append(step_name)
        logger.debug(f"Executing step: {step_name}")
        return await step(**kwargs)

    async def run(self, **kwargs) -> Any:
        """Executes the workflow with the given keyword arguments.

        This method should define the sequence of steps based on the workflow logic.

        Args:
            **kwargs: Arbitrary keyword arguments for the steps.

        Returns:
            Any: The final result of the workflow.
        """
        raise NotImplementedError("Define the run logic in a subclass.")


class ConcreteRAGWorkflow(RAGWorkflow):
    """Concrete implementation of the RAGWorkflow."""

    def __init__(self, embedder: Any, vector_store: Any):
        """
        Initializes the ConcreteRAGWorkflow with necessary components.

        Args:
            embedder (Any): The embedding model instance.
            vector_store (Any): The vector store instance.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        super().__init__()

    @step_config(description="Ingest data from a directory and embed it.")
    async def ingest(self, dirname: str) -> Any:
        logger.debug(f"Ingesting data from directory: {dirname}")

        documents = []
        embeddings = {}

        # List all files in the given directory
        for filename in os.listdir(dirname):
            file_path = os.path.join(dirname, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    # Read the content of the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(content)

                    # Get document embedding
                    embedding = await self.embedder.get_text_embedding_batch([content])
                    embeddings[filename] = embedding[0]  # Assuming embedding returns a list of lists

                    logger.debug(f"Ingested file '{filename}' with content: {content[:30]}...")  # Log the first 30 chars
                except Exception as e:
                    logger.error(f"Failed to read file '{filename}': {e}")

        await self.vector_store.add_embeddings(embeddings)

        # Store ingested document contents
        self.documents = documents  # Store ingested documents for retrieval later
        ingest_result = {"documents": documents, "embeddings": embeddings}
        logger.info(f"Ingest step completed with result: {ingest_result}")
        return ingest_result

    @step_config(description="Retrieve relevant documents based on a query.")
    async def retrieve(self, query: str, index: Any) -> Any:
        if not query:
            raise ValueError("Query must be provided for retrieval.")
        logger.debug(f"Retrieving documents for query: {query}")

        # Simulated retrieval logic: filter documents based on the query
        retrieved_docs = [doc for doc in self.documents if query.lower() in doc.lower()]  # Simple substring match

        retrieved = {"documents": retrieved_docs}
        logger.info(f"Retrieve step completed with retrieved documents: {retrieved}")
        return retrieved

    @step_config(description="Rerank the retrieved documents.")
    async def rerank(self, retrieved: Any) -> Any:
        logger.debug("Reranking documents.")
        await asyncio.sleep(1)  # Simulate rerank operation
        reranked = {"documents": list(reversed(retrieved["documents"]))}
        logger.info(f"Rerank step completed with reranked documents: {reranked}")
        return reranked

    @step_config(description="Synthesize an answer from the reranked documents.")
    async def synthesize(self, reranked: Any) -> Any:
        logger.debug("Synthesizing answer from documents.")
        await asyncio.sleep(1)  # Simulate synthesis
        synthesized_answer = f"Synthesized answer based on documents: {', '.join(reranked['documents'])}"
        logger.info(f"Synthesize step completed with answer: {synthesized_answer}")
        return synthesized_answer

    async def run(self, dirname: Optional[str] = None, query: Optional[str] = None, index: Optional[Any] = None) -> Any:
        if dirname:
            ingest_result = await self.run_step("ingest", dirname=dirname)
        else:
            ingest_result = index

        if query:
            retrieved = await self.run_step("retrieve", query=query, index=ingest_result)
            reranked = await self.run_step("rerank", retrieved=retrieved)
            synthesized = await self.run_step("synthesize", reranked=reranked)
            return synthesized
        else:
            raise ValueError("Query must be provided to run the retrieve, rerank, and synthesize steps.")