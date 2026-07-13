import re

class ArticleCatalogParser:
    def __init__(self):
        pass

    def parse_catalog(self, txt_content):
        parsed_entries = []
        for line in txt_content.splitlines():
            if not line.strip():
                continue
            entry = self.parse_line(line)
            if entry:
                parsed_entries.append(entry)
        return parsed_entries

    def parse_line(self, raw_line):
        original_line = raw_line.strip()
        
        # 1. Extract article number safely
        # Matches leading digits optionally followed by dot, dash, or parenthesis
        match = re.match(r'^\s*(\d+)[\.\-\)]?\s*(.*)', original_line)
        if match:
            article_number = match.group(1)
            content = match.group(2)
        else:
            article_number = None
            content = original_line

        # 2. Extract importance markers
        importance = "regular"
        marker_raw = ""
        
        # Find all occurrences of (*) or (**)
        markers = re.findall(r'\(\*{1,2}\)', content)
        if markers:
            marker_raw = " ".join(markers)
            if "(**)" in markers:
                importance = "even_more_important"
            elif "(*)" in markers:
                importance = "more_important"
            
            # Remove markers from content and clean up whitespace
            content = re.sub(r'\(\*{1,2}\)', '', content).strip()
            
        # 3. Parse author and title
        left_part_raw = None
        right_part_raw = None
        cleaned_title = None
        cleaned_authors = None
        orientation_decision = "unresolved"
        parse_method = "unknown"
        parse_warning = None

        if '/' in content:
            # We treat the first slash as the separator
            parts = content.split('/', 1)
            left_part_raw = parts[0].strip()
            right_part_raw = parts[1].strip()
            
            parse_method = "slash_split"
            
            # If there are multiple slashes, warn the user
            if '/' in right_part_raw:
                parse_warning = "Multiple '/' found; treated first as primary separator."

            # Orientation Heuristics
            # Authors are usually 1-3 words. Titles are usually longer.
            left_words = len(left_part_raw.split())
            right_words = len(right_part_raw.split())

            if left_words <= 3 and right_words > 3:
                orientation_decision = "author_then_title"
                cleaned_authors = left_part_raw
                cleaned_title = right_part_raw
            elif right_words <= 3 and left_words > 3:
                orientation_decision = "title_then_author"
                cleaned_authors = right_part_raw
                cleaned_title = left_part_raw
            else:
                orientation_decision = "unresolved"
                parse_warning = (parse_warning + "; " if parse_warning else "") + "Ambiguous author/title orientation."
                # Do not eagerly map to cleaned fields if unresolved to remain conservative
        else:
            # No reliable separator
            left_part_raw = content
            cleaned_title = content
            orientation_decision = "unresolved"
            parse_method = "raw_fallback"
            parse_warning = "No '/' separator found; treating entire string as title."
            
        return {
            "article_number": article_number,
            "raw_line": original_line,
            "marker_raw": marker_raw,
            "importance": importance,
            "left_part_raw": left_part_raw,
            "right_part_raw": right_part_raw,
            "cleaned_title": cleaned_title,
            "cleaned_authors": cleaned_authors,
            "orientation_decision": orientation_decision,
            "parse_method": parse_method,
            "parse_warning": parse_warning
        }
