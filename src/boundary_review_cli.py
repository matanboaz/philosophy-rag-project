import sys
import os

# Ensure the parent directory is in the python path to import ingestion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_parser import PDFParser
from ingestion.segmenter import BoundarySegmenter
from ingestion.pdf_splitter import PDFSplitter

def main():
    print("=== Phase 1: Preprocessing & Boundary Review ===")
    
    # Define expected paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "raw", "main.pdf")
    titles_path = os.path.join(base_dir, "data", "raw", "titles.txt")
    out_dir = os.path.join(base_dir, "data", "processed", "split_pdfs")
    
    if not os.path.exists(pdf_path) or not os.path.exists(titles_path):
        print(f"Error: Missing input files.")
        print(f"Please ensure the following exist:")
        print(f"1. {pdf_path}")
        print(f"2. {titles_path}")
        sys.exit(1)
        
    # 1. Read Titles
    with open(titles_path, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
        
    print(f"\n[Step 1] Ingesting {pdf_path}...")
    parser = PDFParser(pdf_path)
    pages_text = parser.extract_text(bidi_reorder=True) # RTL handling
    
    print("\n[Step 2] Detecting Boundaries based on titles...")
    segmenter = BoundarySegmenter(titles)
    proposed, unmatched = segmenter.detect_boundaries(pages_text)
    
    # 3. Review Loop (PAUSE FOR REVIEW)
    while True:
        print("\n--- PROPOSED BOUNDARIES ---")
        for i, b in enumerate(proposed):
            print(f"[{i}] {b['title']} (Pages {b['start_page']} - {b['end_page']})")
            
        if unmatched:
            print("\n--- UNMATCHED TITLES ---")
            for ut in unmatched:
                print(f"- {ut}")
                
        print("\nAction Required:")
        print("(A)pprove these boundaries and split PDF")
        print("(E)dit a boundary manually")
        print("(Q)uit without saving")
        
        choice = input("> ").strip().lower()
        
        if choice == 'a':
            print("\n[Step 3] Boundaries Approved. Generating physical split PDFs...")
            splitter = PDFSplitter(pdf_path, out_dir)
            meta = splitter.split_and_save(proposed)
            print(f"\nSuccess! {len(meta)} split PDFs generated in:\n{out_dir}")
            print(f"Metadata saved to: {os.path.join(out_dir, 'articles_metadata.json')}")
            break
        elif choice == 'e':
            try:
                idx = int(input("Enter boundary index to edit: "))
                if 0 <= idx < len(proposed):
                    new_start = int(input(f"New start page (current {proposed[idx]['start_page']}): "))
                    new_end = int(input(f"New end page (current {proposed[idx]['end_page']}): "))
                    proposed[idx]['start_page'] = new_start
                    proposed[idx]['end_page'] = new_end
                    print("Boundary updated.")
                else:
                    print("Invalid index.")
            except ValueError:
                print("Please enter valid numbers.")
        elif choice == 'q':
            print("Aborted by user. No files were indexed or split.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
