import os
from typing import List, Tuple

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridSearch:
    """
    On-prem hybrid search combining BM25 and semantic embeddings via FAISS.

    - BM25: uses rank_bm25 package
    - Semantic: uses sentence-transformers + FAISS
    """

    def __init__(
        self,
        corpus: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.corpus = corpus

        # 日本語用トークナイザ（fugashi：MeCabベース）
        self.tagger = Tagger()
        tokenized = [[word.surface for word in self.tagger(doc)] for doc in corpus]

        # BM25初期化
        self.bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)

        # 意味ベクトル
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            corpus, convert_to_numpy=True, show_progress_bar=True
        )

        # FAISSインデックス構築
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        ids = list(range(len(corpus)))
        self.index.add_with_ids(self.embeddings, ids)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search and return top_k results.

        :param query: the search query
        :param top_k: number of results to return
        :param alpha: weight for semantic score (0 <= alpha <= 1)
        """
        # BM25 scores
        bm_scores = self.bm25.get_scores(query.split())

        # Semantic similarity scores via FAISS
        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, len(self.corpus))
        sem_scores = 1 / (1 + distances[0])  # convert L2 distance to similarity

        # Normalize scores
        bm_min, bm_max = bm_scores.min(), bm_scores.max()
        sem_min, sem_max = sem_scores.min(), sem_scores.max()

        bm_norm = (bm_scores - bm_min) / (bm_max - bm_min + 1e-8)
        sem_norm = (sem_scores - sem_min) / (sem_max - sem_min + 1e-8)

        # Hybrid score
        hybrid_scores = alpha * sem_norm + (1 - alpha) * bm_norm

        # Get top_k
        top_indices = hybrid_scores.argsort()[::-1][:top_k]
        return [(self.corpus[idx], float(hybrid_scores[idx])) for idx in top_indices]


if __name__ == "__main__":
    # Example usage
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "FAISS provides efficient similarity search.",
        "Rank-BM25 is great for keyword-based retrieval.",
        "SentenceTransformers enables semantic embeddings.",
        "Hybrid search combines best of both worlds.",
    ]

    hs = HybridSearch(docs)
    query = "efficient semantic search"
    results = hs.hybrid_search(query, top_k=3, alpha=0.6)

    print("Top results:")
    for text, score in results:
        print(f"Score: {score:.4f} - {text}")
