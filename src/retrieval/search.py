import json
import os
import chromadb
from rank_bm25 import BM25Okapi
import time
from uuid import uuid4

class HybridSearcher:
    _client_cache = {}

    def __init__(self, registry_path, db_path):
        self.registry_path = registry_path
        self.db_path = db_path
        self.chunks = []
        self.bm25 = None
        
        # Prevent RustBindingsAPI teardown bugs by caching the client per path
        if self.db_path not in HybridSearcher._client_cache:
            HybridSearcher._client_cache[self.db_path] = chromadb.PersistentClient(path=self.db_path)
        self.chroma_client = HybridSearcher._client_cache[self.db_path]
        
        # Explicitly configure multilingual embedding model for mixed Hebrew/English retrieval
        from chromadb.utils import embedding_functions
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="philosophy_docs_v2", 
            embedding_function=self.ef
        )
        
        self._load_and_index()
        
    def _load_and_index(self):
        print("Loading chunks from registry...")
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
                
        # BM25 Tokenization (Simple whitespace/punctuation split)
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # ChromaDB Indexing
        if self.collection.count() == 0:
            print(f"Indexing {len(self.chunks)} chunks into ChromaDB...")
            ids = [c["chunk_id"] for c in self.chunks]
            documents = [c["text"] for c in self.chunks]
            metadatas = [{k: v for k, v in c.items() if k != "text"} for c in self.chunks]
            # Batching to avoid issues
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        else:
            print(f"ChromaDB already contains {self.collection.count()} chunks. Skipping index build.")

    def rrf(self, rank_lists, k=60):
        """Reciprocal Rank Fusion"""
        rrf_scores = {}
        for rank_list in rank_lists:
            for rank, item_id in enumerate(rank_list):
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                rrf_scores[item_id] += 1.0 / (k + rank + 1)
        return rrf_scores

    def search(self, query, top_k=5):
        # 1. Lexical Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Sort BM25 results
        lexical_ranked = sorted([
            (self.chunks[i]["chunk_id"], bm25_scores[i]) 
            for i in range(len(self.chunks)) if bm25_scores[i] > 0
        ], key=lambda x: x[1], reverse=True)
        lexical_ids = [x[0] for x in lexical_ranked]
        
        # 2. Dense Search
        dense_results = self.collection.query(query_texts=[query], n_results=len(self.chunks))
        dense_ids = dense_results["ids"][0]
        dense_distances = dense_results["distances"][0]
        # In Chroma, lower distance = better score (L2 default). We rank by distance ascending.
        dense_ranked = [(dense_ids[i], dense_distances[i]) for i in range(len(dense_ids))]
        
        # 3. Fusion (RRF)
        fused_scores = self.rrf([lexical_ids, dense_ids])
        
        # 4. Final Ranking (Mocking Reranker Fallback - relying on RRF)
        final_ranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
        
        # 5. Build Result Artifact
        chunk_lookup = {c["chunk_id"]: c for c in self.chunks}
        lexical_lookup = dict(lexical_ranked)
        dense_lookup = dict(dense_ranked)
        
        results = []
        for rank, cid in enumerate(final_ranked_ids):
            chunk = chunk_lookup[cid]
            results.append({
                "rank": rank + 1,
                "chunk_id": cid,
                "article_id": chunk.get("article_id", "N/A"),
                "article_title": chunk["article_title"],
                "global_page_num": chunk["global_page_num"],
                "local_page_num": chunk.get("local_page_num", 1),
                "char_start": chunk.get("char_start", -1),
                "char_end": chunk.get("char_end", -1),
                "line_start": chunk.get("line_start", -1),
                "line_end": chunk.get("line_end", -1),
                "fused_rrf_score": fused_scores[cid],
                "lexical_score": lexical_lookup.get(cid, 0.0),
                "dense_distance": dense_lookup.get(cid, 999.0),
                "text": chunk["text"]
            })
            
        return results

    def close(self):
        """Release underlying file locks so databases can be safely deleted."""
        if hasattr(self, 'chroma_client') and self.chroma_client is not None:
            try:
                self.chroma_client.close()
                if hasattr(self.chroma_client, 'clear_system_cache'):
                    self.chroma_client.clear_system_cache()
            except Exception:
                pass
                
            if self.db_path in HybridSearcher._client_cache:
                del HybridSearcher._client_cache[self.db_path]
                
            self.chroma_client = None
