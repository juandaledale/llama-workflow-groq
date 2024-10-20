# embedder.py
from typing import List


class Embedder:

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        # Implement your text embedding logic here
        return [[0.0] * 768 for _ in texts]

    async def aget_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        # Asynchronous embedding logic
        return self.get_text_embedding_batch(texts, show_progress)