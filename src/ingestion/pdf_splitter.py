import fitz
import os
import json
import re

class PDFSplitter:
    def __init__(self, input_pdf_path, output_dir):
        self.input_pdf_path = input_pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def split_and_save(self, boundary_proposals):
        """
        Creates physical split PDFs based on boundary proposals and saves metadata.
        NOTE ON PAGE NUMBERING: 
        The manifest and boundary_proposals use 1-based page indices for human readability.
        PyMuPDF (fitz) internally uses 0-based indexing. This method explicitly handles the -1 offset.
        """
        manifest = []
        summary = {
            "total_catalog_entries": len(boundary_proposals),
            "total_successfully_split": 0,
            "total_skipped": 0,
            "skipped_reasons": {}
        }
        
        doc = fitz.open(self.input_pdf_path)
        
        for prop in boundary_proposals:
            article_number = prop.get('article_number')
            title = prop.get('cleaned_title')
            start_page = prop.get('proposed_start_page')
            end_page = prop.get('proposed_end_page')
            
            output_filename = None
            split_status = "success"
            warning = prop.get('warning')
            
            actual_start_page = None
            actual_end_page = None
            range_adjusted = False
            
            if start_page is None or end_page is None:
                split_status = "skipped_unmatched"
                warning = (warning + "; " if warning else "") + "Missing start or end page."
            elif end_page < start_page:
                split_status = "skipped_invalid_range"
                warning = (warning + "; " if warning else "") + f"End page ({end_page}) is before start page ({start_page})."
            else:
                # 1-based to 0-based conversion for fitz
                fitz_start = start_page - 1
                fitz_end = end_page - 1
                
                actual_start_page = start_page
                actual_end_page = end_page
                
                # Bounds check explicitly tracked
                if fitz_start < 0: 
                    fitz_start = 0
                    actual_start_page = 1
                    range_adjusted = True
                if fitz_end >= len(doc): 
                    fitz_end = len(doc) - 1
                    actual_end_page = len(doc)
                    range_adjusted = True
                    
                if range_adjusted:
                    split_status = "success_range_adjusted"
                    warning = (warning + "; " if warning else "") + f"Proposed range exceeded bounds; clipped to {actual_start_page}-{actual_end_page}."
                
                # Create safe filename
                safe_title = "fallback"
                if title:
                    safe_title = re.sub(r'[^\w\s]', '', title).strip()
                    safe_title = safe_title.replace(' ', '_')
                if not safe_title:
                    safe_title = "fallback"
                    
                safe_name = f"art_{article_number}_{safe_title}" if article_number else f"art_{safe_title}"
                output_filename = f"{safe_name}.pdf"
                out_path = os.path.join(self.output_dir, output_filename)
                
                # Prevent silent overwriting
                counter = 2
                while os.path.exists(out_path):
                    output_filename = f"{safe_name}_dup{counter}.pdf"
                    out_path = os.path.join(self.output_dir, output_filename)
                    if f"_dup{counter}" not in (warning or ""):
                        warning = (warning + "; " if warning else "") + f"Filename collision resolved by appending _dup{counter}."
                    counter += 1
                
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=fitz_start, to_page=fitz_end)
                new_doc.save(out_path)
                new_doc.close()
                
            manifest_entry = {
                "article_number": article_number,
                "raw_line": prop.get('raw_line'),
                "importance": prop.get('importance'),
                "orientation_decision": prop.get('orientation_decision'),
                "cleaned_title": title,
                "cleaned_authors": prop.get('cleaned_authors'),
                "proposed_start_page": start_page,
                "proposed_end_page": end_page,
                "actual_start_page": actual_start_page,
                "actual_end_page": actual_end_page,
                "range_adjusted": range_adjusted,
                "output_pdf_filename": output_filename,
                "split_status": split_status,
                "warning": warning
            }
            manifest.append(manifest_entry)
            
            if split_status.startswith("success"):
                summary["total_successfully_split"] += 1
            else:
                summary["total_skipped"] += 1
                summary["skipped_reasons"][split_status] = summary["skipped_reasons"].get(split_status, 0) + 1
                
        doc.close()
        
        # Save manifest and summary
        meta_path = os.path.join(self.output_dir, "articles_manifest.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({"manifest": manifest, "summary": summary}, f, ensure_ascii=False, indent=2)
            
        return manifest, summary
