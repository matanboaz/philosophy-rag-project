import sys
import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.ingestion.txt_parser import ArticleCatalogParser
from src.ingestion.segmenter import BoundaryProposer
from src.ingestion.pdf_splitter import PDFSplitter
from src.ingestion.pdf_parser import PDFParser

def run_pipeline():
    # 1. Define Paths
    pdf_path = os.path.join(base_dir, "data", "raw", "main.pdf")
    txt_path = os.path.join(base_dir, "data", "raw", "articles.txt")
    out_dir = os.path.join(base_dir, "data", "processed", "split_pdfs")
    
    if not os.path.exists(pdf_path) or not os.path.exists(txt_path):
        print(f"Error: Missing input files. Ensure {pdf_path} and {txt_path} exist.")
        return

    # 2. Parse TXT
    print("Parsing articles.txt...")
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    
    catalog_parser = ArticleCatalogParser()
    catalog_entries = catalog_parser.parse_catalog(txt_content)
    print(f"Parsed {len(catalog_entries)} entries.")

    # 3. Extract PDF Text & Propose Boundaries
    print("Extracting text from PDF (this may take a moment)...")
    pdf_parser = PDFParser(pdf_path)
    pages_text = pdf_parser.extract_text(bidi_reorder=False)
    
    print("Proposing boundaries...")
    proposer = BoundaryProposer(catalog_entries)
    boundary_proposals = proposer.propose_boundaries(pages_text)

    # 4. Split and Save
    print("Executing physical PDF splits...")
    splitter = PDFSplitter(pdf_path, out_dir)
    manifest, summary = splitter.split_and_save(boundary_proposals)
    
    print("\n=== PIPELINE COMPLETE ===")
    print(f"Total Entries: {summary['total_catalog_entries']}")
    print(f"Successfully Split: {summary['total_successfully_split']}")
    print(f"Skipped: {summary['total_skipped']}")
    print(f"Manifest saved to: {os.path.join(out_dir, 'articles_manifest.json')}")

if __name__ == "__main__":
    run_pipeline()
