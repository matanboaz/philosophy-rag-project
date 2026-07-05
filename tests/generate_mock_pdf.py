import os
import fitz

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    pdf_path = os.path.join(raw_dir, "main.pdf")
    titles_path = os.path.join(raw_dir, "titles.txt")
    
    # 1. Create a mock PDF with 4 pages
    doc = fitz.open()
    
    # Page 1: Introduction (No explicit title)
    page1 = doc.new_page()
    page1.insert_text((50, 150), "This is the introduction to the book. (הקדמה לספר)")
    
    # Page 2: Article 1
    page2 = doc.new_page()
    page2.insert_text((50, 150), "The Concept of Time in Maimonides")
    page2.insert_text((50, 180), "By Author A. Here is the text discussing how Rambam views time...")
    
    # Page 3: Article 1 continued
    page3 = doc.new_page()
    page3.insert_text((50, 150), "More text about Maimonides. Time is related to motion. (הזמן קשור לתנועה אצל הרמבם)")
    
    # Page 4: Article 2
    page4 = doc.new_page()
    page4.insert_text((50, 150), "Spinoza on Free Will")
    page4.insert_text((50, 180), "By Author B. Philosophy text about Spinoza and human choices... (שפינוזה כותב על חופש הרצון והבחירה)")
    
    doc.save(pdf_path)
    doc.close()
    
    # 2. Create titles.txt (with one unmatched title to prove it works)
    with open(titles_path, 'w', encoding='utf-8') as f:
        f.write("The Concept of Time in Maimonides\n")
        f.write("Spinoza on Free Will\n")
        f.write("Unmatched Article on Ethics\n")
        
    print(f"Mock data created successfully in {raw_dir}")

if __name__ == "__main__":
    main()
