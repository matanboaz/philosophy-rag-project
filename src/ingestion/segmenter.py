import re

class BoundaryProposer:
    def __init__(self, catalog_entries):
        """
        catalog_entries is a list of dicts parsed by ArticleCatalogParser.
        """
        self.catalog_entries = catalog_entries

    def propose_boundaries(self, pages_text):
        """
        pages_text is a dict of {page_num: text}
        Returns a list of proposal dicts.
        """
        proposals = []
        
        for entry in self.catalog_entries:
            title = entry.get("cleaned_title", "")
            author = entry.get("cleaned_authors", "")
            
            left_raw = entry.get("left_part_raw", "")
            right_raw = entry.get("right_part_raw", "")
            orientation = entry.get("orientation_decision", "unresolved")
            
            found_start_page = None
            match_score = 0
            matching_method = "none"
            
            for page_num, text in pages_text.items():
                clean_text = re.sub(r'[^\w\s]', '', text).strip()
                
                # If we have a resolved orientation
                if orientation in ["author_then_title", "title_then_author"] and title:
                    clean_title = re.sub(r'[^\w\s]', '', title).strip()
                    clean_author = re.sub(r'[^\w\s]', '', author).strip() if author else ""
                    
                    title_match = clean_title and clean_title in clean_text
                    author_match = clean_author and clean_author in clean_text
                    
                    if title_match and author_match:
                        found_start_page = page_num
                        matching_method = "exact_title_and_author"
                        match_score = 100
                        break
                    elif title_match:
                        found_start_page = page_num
                        matching_method = "exact_title_only"
                        match_score = 80
                        break
                    elif author_match:
                        found_start_page = page_num
                        matching_method = "exact_author_only"
                        match_score = 50
                        break
                else:
                    # Unresolved or raw fallback -> do cautious dual-sided search
                    c_left = re.sub(r'[^\w\s]', '', left_raw).strip() if left_raw else ""
                    c_right = re.sub(r'[^\w\s]', '', right_raw).strip() if right_raw else ""
                    
                    left_match = c_left and c_left in clean_text
                    right_match = c_right and c_right in clean_text
                    
                    if left_match and right_match:
                        found_start_page = page_num
                        matching_method = "ambiguous_dual_match"
                        match_score = 90
                        break
                    elif left_match or right_match:
                        found_start_page = page_num
                        matching_method = "ambiguous_single_match"
                        match_score = 60
                        break
                    
            proposals.append({
                "article_number": entry.get("article_number"),
                "raw_line": entry.get("raw_line"),
                "cleaned_title": title,
                "cleaned_authors": author,
                "importance": entry.get("importance"),
                "proposed_start_page": found_start_page,
                "proposed_end_page": None,
                "title_match_score": 100 if "title" in matching_method else 0,
                "author_match_score": 100 if "author" in matching_method else 0,
                "matching_method": matching_method,
                "warning": "No match found in PDF." if not found_start_page else None
            })
            
        # Determine end pages based on sequential start pages
        valid_proposals = [p for p in proposals if p["proposed_start_page"] is not None]
        valid_proposals.sort(key=lambda x: x["proposed_start_page"])
        
        for i, prop in enumerate(valid_proposals):
            if i + 1 < len(valid_proposals):
                end_page = valid_proposals[i+1]["proposed_start_page"] - 1
                if end_page < prop["proposed_start_page"]:
                    end_page = prop["proposed_start_page"]
                prop["proposed_end_page"] = end_page
            else:
                prop["proposed_end_page"] = max(pages_text.keys()) if pages_text else prop["proposed_start_page"]
                
        # Merge back proposed end pages into the original list order
        for p in proposals:
            if p["proposed_start_page"] is not None:
                vp = next(x for x in valid_proposals if x["raw_line"] == p["raw_line"])
                p["proposed_end_page"] = vp["proposed_end_page"]
                
        return proposals

class BoundarySegmenter:
    def __init__(self, titles):
        self.titles = [t.strip() for t in titles if t.strip()]
        self.boundaries = []
        self.unmatched_titles = []

    def detect_boundaries(self, pages_text):
        """
        Takes {page_num: text} and a list of titles.
        Returns proposed boundaries and unmatched titles.
        """
        detected = {}
        
        for title in self.titles:
            found = False
            for page_num, text in pages_text.items():
                # Strip punctuation and spaces for robust matching (works for Hebrew/English)
                clean_title = re.sub(r'[^\w\s]', '', title).strip()
                clean_text = re.sub(r'[^\w\s]', '', text).strip()
                
                if clean_title and clean_title in clean_text:
                    detected[title] = page_num
                    found = True
                    break
                    
            if not found:
                self.unmatched_titles.append(title)
                
        # Sort detected by page_num to find end_pages
        sorted_detected = sorted(detected.items(), key=lambda x: x[1])
        
        for i, (title, start_page) in enumerate(sorted_detected):
            if i + 1 < len(sorted_detected):
                end_page = sorted_detected[i+1][1] - 1
                if end_page < start_page: 
                    end_page = start_page # Safety check
            else:
                end_page = max(pages_text.keys()) if pages_text else start_page
                
            self.boundaries.append({
                "title": title,
                "start_page": start_page,
                "end_page": end_page
            })
            
        return self.boundaries, self.unmatched_titles
