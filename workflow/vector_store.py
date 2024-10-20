from typing import Any, Dict, List, Optional

class VectorStore:
    def add_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        # Implement logic to add embeddings to the store
        pass

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        # Implement query logic to retrieve top_k relevant document IDs
        return ["doc1", "doc2", "doc3", "doc4", "doc5"]