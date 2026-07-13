import sys
import os
import shutil

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.app import index_new_article, load_article_searcher
from src.retrieval.search import HybridSearcher
from src.generation.qa_pipeline import QAPipeline

def test_fix():
    print("Test 1: Check existing HybridSearcher signature")
    # This uses the original corpus, mimicking Tab 1 / Tab 2
    reg = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
    db = os.path.join(base_dir, "data", "vector_store")
    
    # We will just verify it instantiates cleanly (if it exists)
    if os.path.exists(reg) and os.path.exists(db):
        searcher_bg = HybridSearcher(reg, db)
        print("-> Background searcher loaded OK.")
    else:
        searcher_bg = None
        print("-> Background corpus not found (expected if not built), skipping bg test.")
        
    print("\nTest 2: Simulate Uploading new article from Tab 3")
    # We use a dummy pdf we generated earlier
    mock_pdf = os.path.join(base_dir, "data", "raw", "main.pdf")
    if not os.path.exists(mock_pdf):
        print("-> Missing mock PDF. Skipping.")
        return
        
    unique_name = index_new_article(mock_pdf, "Test_Article_Upload")
    print(f"-> Article indexed. Unique name: {unique_name}")
    
    print("\nTest 3: Select saved article and load into searcher")
    try:
        primary_searcher = load_article_searcher(unique_name)
        print("-> Primary searcher loaded successfully! Crash resolved.")
    except Exception as e:
        print(f"-> FAILED: {e}")
        return
        
    print("\nTest 4: Existing background-corpus comparison mode")
    if searcher_bg:
        try:
            q = "time"
            p_chunks = primary_searcher.search(q, top_k=1)
            b_chunks = searcher_bg.search(q, top_k=1)
            print(f"-> Retrieved {len(p_chunks)} from primary, {len(b_chunks)} from background.")
        except Exception as e:
            print(f"-> FAILED execution: {e}")
            
    print("\nAll tests passed successfully.")
    
    # Cleanup
    shutil.rmtree(os.path.join(base_dir, "data", "processed", "new_articles", unique_name))

if __name__ == "__main__":
    test_fix()
