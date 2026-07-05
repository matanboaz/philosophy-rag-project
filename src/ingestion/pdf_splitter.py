import fitz
import os
import json

class PDFSplitter:
    def __init__(self, input_pdf_path, output_dir):
        self.input_pdf_path = input_pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def split_and_save(self, approved_boundaries):
        """
        Creates physical split PDFs based on approved boundaries and saves metadata.
        """
        results = []
        doc = fitz.open(self.input_pdf_path)
        
        for i, boundary in enumerate(approved_boundaries):
            title = boundary['title']
            # fitz uses 0-indexed pages
            start_page = boundary['start_page'] - 1 
            end_page = boundary['end_page'] - 1
            
            # Ensure valid bounds
            if start_page < 0: start_page = 0
            if end_page >= len(doc): end_page = len(doc) - 1
            
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            
            # Create a safe filename (removes Hebrew punctuation issues)
            safe_title = "".join([c for c in title if c.isalnum() or c==' ']).rstrip()
            safe_title = safe_title.replace(' ', '_')
            if not safe_title:
                safe_title = f"article_{i+1}"
                
            out_path = os.path.join(self.output_dir, f"{safe_title}.pdf")
            new_doc.save(out_path)
            new_doc.close()
            
            metadata = {
                "article_id": f"art_{i+1}",
                "title": title,
                "source_pdf": self.input_pdf_path,
                "start_page": boundary['start_page'],
                "end_page": boundary['end_page'],
                "split_pdf_path": out_path,
                "approval_status": "approved"
            }
            results.append(metadata)
            
        doc.close()
        
        # Save metadata artifact
        meta_path = os.path.join(self.output_dir, "articles_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
