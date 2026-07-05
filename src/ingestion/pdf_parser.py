import fitz
from bidi.algorithm import get_display

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_text(self, bidi_reorder=True):
        """
        Extracts text page by page.
        Applies bidi reordering for Hebrew text if bidi_reorder is True.
        Returns a dict of {page_num: text}.
        """
        pages_text = {}
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text = page.get_text("text")
            
            if bidi_reorder:
                # Apply bidi reordering line by line for RTL Hebrew
                lines = text.split('\n')
                reordered_lines = [get_display(line) for line in lines]
                text = '\n'.join(reordered_lines)
                
            # 1-indexed pages for human review
            pages_text[page_num + 1] = text
            
        return pages_text
