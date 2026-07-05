import re

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
