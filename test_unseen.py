import os
import sys
import json
base_dir = r"c:\Users\SZMC\Documents\philosophy_rag_project"
sys.path.append(base_dir)

# Mock Streamlit to prevent errors when importing app
import streamlit as st
class MockST:
    def __init__(self):
        self.session_state = {}
    def markdown(self, text, **kwargs):
        print("\n--- ST.MARKDOWN ---")
        print(text)
    def error(self, text, **kwargs):
        print("ST.ERROR:", text)
    def success(self, text, **kwargs):
        print("ST.SUCCESS:", text)
    def warning(self, text, **kwargs):
        print("ST.WARNING:", text)
    def info(self, text, **kwargs):
        print("ST.INFO:", text)
    def spinner(self, text, **kwargs):
        from contextlib import contextmanager
        @contextmanager
        def spinner_mock():
            print("SPINNER:", text)
            yield
        return spinner_mock()
sys.modules['streamlit'] = MockST()

from src.app import index_new_article, load_article_searcher, render_answer_with_citations
from src.generation.qa_pipeline import QAPipeline

pdf_path = os.path.join(base_dir, "data", "raw", "unseen_mock.pdf")

print("1. Indexing new article...")
unique_name = index_new_article(pdf_path, "unseen_mock.pdf")
print("Indexed as:", unique_name)

print("\n2. Checking Registry...")
registry_path = os.path.join(base_dir, "data", "processed", "new_articles", unique_name, "registry.jsonl")
with open(registry_path, "r", encoding="utf-8") as f:
    first_chunk = json.loads(f.readline())
    print("Example Chunk Keys:", list(first_chunk.keys()))
    print("Chunk ID:", first_chunk["chunk_id"])
    print("Char Start/End:", first_chunk.get("char_start"), first_chunk.get("char_end"))
    print("Line Start/End:", first_chunk.get("line_start"), first_chunk.get("line_end"))
    print("Global Page:", first_chunk.get("global_page_num"))

print("\n3. Testing Retrieval...")
searcher = load_article_searcher(unique_name)
query = "מה משמעות הפילוסופיה?"
chunks = searcher.search(query, top_k=2)
print("Retrieved chunks:", [c["chunk_id"] for c in chunks])

print("\n4. Testing QA Pipeline...")
pipeline = QAPipeline(backend_strategy="dicta")
# Mock the LLM to skip actually running Dicta just to verify formatting
# Wait, I want to see if the LLM output can cite chunks correctly.
# Dicta might be slow, let's just run it!
result = pipeline.execute_comparison_qa(query, primary_chunks=chunks, reference_chunks=[], mode="compare", budget=30)
ans_data = result["final_parsed_answer"]
print("\nLLM Output JSON:", json.dumps(ans_data, indent=2, ensure_ascii=False))

print("\n5. Testing Render Citation...")
if "answers" in ans_data:
    for a in ans_data["answers"]:
        matched = render_answer_with_citations(a.get("answer", ""), a.get("citations", []), chunks)
        print("Matched Chunks for Viewer:", list(matched.keys()))
