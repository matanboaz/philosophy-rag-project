import sys
import os
import json
from datetime import datetime

# Setup paths
src_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)

from retrieval.chunker import ChunkRegistryBuilder
from retrieval.search import HybridSearcher

def main():
    print("=== Phase 2 Validation: Indexing & Retrieval ===")
    
    meta_path = os.path.join(base_dir, "data", "processed", "split_pdfs", "articles_metadata.json")
    registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
    db_path = os.path.join(base_dir, "data", "vector_store")
    queries_path = os.path.join(base_dir, "tests", "sample_queries.json")
    logs_dir = os.path.join(base_dir, "data", "processed", "retrieval_logs")
    
    os.makedirs(logs_dir, exist_ok=True)
    
    # 1. Build Chunks
    print("\n[Step 1] Executing Chunking...")
    chunker = ChunkRegistryBuilder(meta_path, registry_path)
    chunker.build_registry()
    
    # 2. Build Indices
    print("\n[Step 2] Initializing Indices (Vector & Lexical)...")
    searcher = HybridSearcher(registry_path, db_path)
    
    # 3. Read Queries
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        sys.exit(1)
        
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    print(f"\n[Step 3] Running Validation on {len(queries)} queries...")
    
    validation_results = []
    
    for q_idx, q in enumerate(queries):
        query_text = q["query"]
        print(f"\nQuery {q_idx + 1}: {query_text}")
        
        results = searcher.search(query_text, top_k=3)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_text": query_text,
            "retrieved_chunk_ids": [r["chunk_id"] for r in results],
            "final_top_k_ranking": results
        }
        validation_results.append(log_entry)
        
        for r in results:
            print(f"  [{r['rank']}] {r['chunk_id']} (RRF: {r['fused_rrf_score']:.4f}) -> {r['text'][:60]}...")
            
    # Save artifacts
    validation_artifact_path = os.path.join(logs_dir, "retrieval_validation_results.json")
    with open(validation_artifact_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccess! Validation artifacts saved to: {validation_artifact_path}")

if __name__ == "__main__":
    main()
