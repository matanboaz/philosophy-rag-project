import sys
import os
import json
from datetime import datetime

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.retrieval.search import HybridSearcher
from src.generation.qa_pipeline import QAPipeline

def main():
    print("=== Phase 3 Validation: Answer Generation ===")
    
    registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
    db_path = os.path.join(base_dir, "data", "vector_store")
    batch_path = os.path.join(base_dir, "tests", "batch_sample.json")
    logs_dir = os.path.join(base_dir, "data", "processed", "generation_logs")
    
    os.makedirs(logs_dir, exist_ok=True)
    
    print("\n[Step 1] Loading Indices and LLM Client...")
    searcher = HybridSearcher(registry_path, db_path)
    
    # Read backend strategy from environment, fallback to 'dicta'
    backend_strategy = os.getenv("LLM_BACKEND_STRATEGY", "dicta")
    pipeline = QAPipeline(backend_strategy=backend_strategy)
    
    if not os.path.exists(batch_path):
        print(f"Error: {batch_path} missing.")
        sys.exit(1)
        
    with open(batch_path, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
        
    batch_guidelines = batch_data.get("batch_guidelines")
    queries = batch_data.get("queries", [])
    
    print(f"\n[Step 2] Processing {len(queries)} queries with Word Budgets and Guidelines...")
    
    logs = []
    
    for q_idx, q in enumerate(queries):
        query_text = q["query"]
        q_guidelines = q.get("question_guidelines")
        budget = q.get("word_budget")
        
        print(f"\nQuery: {query_text} (Budget: {budget})")
        
        # 1. Retrieve
        chunks = searcher.search(query_text, top_k=3)
        
        # 2. Generate
        g_map = {"batch": batch_guidelines, "question": q_guidelines}
        result = pipeline.execute_qa(query_text, chunks, guidelines=g_map, budget=budget)
        
        logs.append(result)
        
        print(f"Warnings: {result['warnings']}")
        print(f"Applied Guidelines: {list(result['applied_guidelines_map'].keys())}")
        
    log_path = os.path.join(logs_dir, "generation_validation_results.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccess! Validation artifacts saved to: {log_path}")

if __name__ == "__main__":
    main()
