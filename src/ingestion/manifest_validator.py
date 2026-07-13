import os
import json
import csv
import fitz
from pathlib import Path

class ManifestValidator:
    def __init__(self, manifest_path, split_pdf_dir, source_pdf_path):
        self.manifest_path = manifest_path
        self.split_pdf_dir = split_pdf_dir
        self.source_pdf_path = source_pdf_path
        self.source_total_pages = 0
        if os.path.exists(self.source_pdf_path):
            try:
                with fitz.open(self.source_pdf_path) as doc:
                    self.source_total_pages = len(doc)
            except:
                pass

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If the JSON is wrapped in an object with a "manifest" key, extract it
            if isinstance(data, dict) and "manifest" in data:
                return data["manifest"]
            return data

    def validate(self, check_first_page=False, output_format=None, output_path=None):
        entries = self.load_manifest()
        report = []
        
        for entry in entries:
            article_number = str(entry.get("article_number", ""))
            title = entry.get("title") or entry.get("cleaned_title", "")
            start_page = entry.get("start_page") if entry.get("start_page") is not None else entry.get("proposed_start_page")
            end_page = entry.get("end_page") if entry.get("end_page") is not None else entry.get("proposed_end_page")
            filename = entry.get("filename") or entry.get("output_pdf_filename")
            
            base_rec = {
                "article_number": article_number,
                "title": title,
                "filename": filename,
                "start_page": start_page,
                "end_page": end_page,
                "expected_page_count": 0,
                "actual_page_count": 0,
                "status": "valid",
                "warning": ""
            }
            
            if start_page is None or end_page is None:
                base_rec.update({"status": "invalid", "warning": "Missing start_page or end_page"})
                report.append(base_rec)
                continue
                
            expected_count = (end_page - start_page) + 1
            base_rec["expected_page_count"] = expected_count
            
            # Strict bounds validation
            if start_page < 1:
                base_rec.update({"status": "invalid", "warning": f"start_page {start_page} < 1"})
                report.append(base_rec)
                continue
            if end_page < start_page:
                base_rec.update({"status": "invalid", "warning": f"end_page {end_page} < start_page {start_page}"})
                report.append(base_rec)
                continue
            if self.source_total_pages > 0 and end_page > self.source_total_pages:
                base_rec.update({"status": "invalid", "warning": f"end_page {end_page} > source bounds {self.source_total_pages}"})
                report.append(base_rec)
                continue

            target_path = os.path.join(self.split_pdf_dir, filename) if filename else None
            
            if not target_path or not os.path.exists(target_path):
                base_rec.update({"status": "missing", "warning": f"File does not exist: {filename}"})
                report.append(base_rec)
                continue
                
            # Check actual page count
            try:
                with fitz.open(target_path) as doc:
                    actual_count = len(doc)
                    base_rec["actual_page_count"] = actual_count
                    
                    first_page_text = ""
                    if check_first_page and actual_count > 0:
                        first_page_text = doc[0].get_text() or ""
            except Exception as e:
                base_rec.update({"status": "invalid", "warning": f"Could not read PDF: {str(e)}"})
                report.append(base_rec)
                continue
                
            if actual_count != expected_count:
                base_rec.update({"status": "suspicious", "warning": f"Page count mismatch. Expected: {expected_count}, Actual: {actual_count}"})
                report.append(base_rec)
                continue
                
            if check_first_page:
                # Basic heuristic check
                text_lower = first_page_text.lower()
                title_lower = str(title).lower()
                art_num_str = str(article_number)
                
                words = title_lower.split()
                matched_words = sum(1 for w in words if w in text_lower and len(w) > 2)
                
                looks_valid = (art_num_str in text_lower) or (matched_words > 0)
                if not looks_valid:
                    base_rec.update({"status": "suspicious", "warning": "First page text lacks title/number match"})
            
            report.append(base_rec)

        if output_format and output_path:
            if output_format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            elif output_format == "csv":
                with open(output_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=report[0].keys())
                    writer.writeheader()
                    writer.writerows(report)
                
        return report

    def regenerate_from_manifest(self, dry_run=False, overwrite=False):
        """
        Regenerates all split PDFs sequentially from the original source PDF based purely on manifest boundaries.
        dry_run: if True, do not write files, just return the list of operations.
        overwrite: if False, skip regeneration if the target file already exists.
        """
        if not os.path.exists(self.source_pdf_path):
            raise FileNotFoundError(f"Source PDF not found: {self.source_pdf_path}")
            
        entries = self.load_manifest()
        
        if not dry_run:
            os.makedirs(self.split_pdf_dir, exist_ok=True)
            
        operations = []
        
        try:
            doc = fitz.open(self.source_pdf_path)
            total_pages = len(doc)
            
            for entry in entries:
                article_number = entry.get("article_number")
                title = entry.get("title") or entry.get("cleaned_title")
                start_page = entry.get("start_page") if entry.get("start_page") is not None else entry.get("proposed_start_page")
                end_page = entry.get("end_page") if entry.get("end_page") is not None else entry.get("proposed_end_page")
                filename = entry.get("filename") or entry.get("output_pdf_filename")
                
                op = {
                    "filename": filename,
                    "action": "skip",
                    "reason": "",
                    "pages": []
                }
                
                if start_page is None or end_page is None or not filename:
                    op["reason"] = "Missing start, end, or filename in manifest"
                    operations.append(op)
                    continue
                    
                target_path = os.path.join(self.split_pdf_dir, filename)
                if os.path.exists(target_path) and not overwrite:
                    op["reason"] = "File exists and overwrite=False"
                    operations.append(op)
                    continue
                
                start_idx = start_page - 1
                end_idx = end_page - 1
                
                if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                    op["reason"] = f"Invalid bounds [{start_page}-{end_page}] (Total: {total_pages})"
                    operations.append(op)
                    continue
                    
                op["action"] = "regenerate"
                op["pages"] = list(range(start_idx, end_idx + 1))
                
                if not dry_run:
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)
                    new_doc.save(target_path)
                    new_doc.close()
                
                operations.append(op)
                
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Error processing source PDF: {e}")
                
        return operations

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Manifest Validator and Regenerator")
    parser.add_argument("--manifest-path", required=True, help="Path to articles_manifest.json")
    parser.add_argument("--split-pdf-dir", required=True, help="Directory containing split PDFs")
    parser.add_argument("--source-pdf-path", required=True, help="Path to original anthology PDF")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate manifest entries against files")
    val_parser.add_argument("--check-first-page", action="store_true", help="Extract first page text to check against title")
    val_parser.add_argument("--output-format", choices=["json", "csv"], help="Format to save report")
    val_parser.add_argument("--output-path", help="Path to save the validation report")
    
    # Regenerate command
    regen_parser = subparsers.add_parser("regenerate", help="Regenerate split PDFs from source")
    regen_parser.add_argument("--dry-run", action="store_true", help="Print planned operations without writing files")
    regen_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        validator = ManifestValidator(
            manifest_path=args.manifest_path,
            split_pdf_dir=args.split_pdf_dir,
            source_pdf_path=args.source_pdf_path
        )
        
        if args.command == "validate":
            report = validator.validate(
                check_first_page=args.check_first_page,
                output_format=args.output_format,
                output_path=args.output_path
            )
            
            valid = sum(1 for r in report if r['status'] == 'valid')
            invalid = sum(1 for r in report if r['status'] == 'invalid')
            missing = sum(1 for r in report if r['status'] == 'missing')
            suspicious = sum(1 for r in report if r['status'] == 'suspicious')
            
            print("=== Validation Summary ===")
            print(f"Total entries: {len(report)}")
            print(f"Valid: {valid}")
            print(f"Invalid: {invalid}")
            print(f"Missing: {missing}")
            print(f"Suspicious: {suspicious}")
            
            if invalid > 0:
                print("\nError: Found invalid entries.")
                sys.exit(2)
                
        elif args.command == "regenerate":
            operations = validator.regenerate_from_manifest(
                dry_run=args.dry_run,
                overwrite=args.overwrite
            )
            
            regen_count = sum(1 for op in operations if op["action"] == "regenerate")
            skip_count = sum(1 for op in operations if op["action"] == "skip")
            
            print(f"=== Regeneration {'Dry Run ' if args.dry_run else ''}Summary ===")
            print(f"Total operations: {len(operations)}")
            print(f"Regenerated: {regen_count}")
            print(f"Skipped: {skip_count}")
            
            if args.dry_run:
                print("\nDetailed Operations:")
                for op in operations:
                    if op["action"] == "regenerate":
                        print(f"[REGENERATE] {op['filename']} (Pages: {len(op['pages'])})")
                    else:
                        print(f"[SKIP] {op['filename']} - {op['reason']}")
                        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
