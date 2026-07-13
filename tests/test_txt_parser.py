import sys
import os
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.ingestion.txt_parser import ArticleCatalogParser

def test_numbered_line_parsing():
    parser = ArticleCatalogParser()
    entry = parser.parse_line("14. Author / Title Of A Long Article")
    assert entry["article_number"] == "14"
    assert entry["orientation_decision"] == "author_then_title"
    assert entry["cleaned_authors"] == "Author"
    assert entry["cleaned_title"] == "Title Of A Long Article"

def test_marker_parsing():
    parser = ArticleCatalogParser()
    
    # Even more important
    entry = parser.parse_line("1. Author / Title (**)")
    assert entry["importance"] == "even_more_important"
    assert entry["marker_raw"] == "(**)"
    
    # More important
    entry2 = parser.parse_line("2. Author / Title (*)")
    assert entry2["importance"] == "more_important"
    assert entry2["marker_raw"] == "(*)"
    
    # Both markers -> highest priority
    entry3 = parser.parse_line("3. Author / Title (*) (**)")
    assert entry3["importance"] == "even_more_important"
    assert entry3["marker_raw"] == "(*) (**)"
    
    # No markers
    entry4 = parser.parse_line("4. Author / Title")
    assert entry4["importance"] == "regular"

def test_slash_based_parsing():
    parser = ArticleCatalogParser()
    
    # Author then Title
    entry = parser.parse_line("1. John Doe / A Philosophy of Time and Space")
    assert entry["cleaned_authors"] == "John Doe"
    assert entry["cleaned_title"] == "A Philosophy of Time and Space"
    assert entry["orientation_decision"] == "author_then_title"
    assert entry["parse_method"] == "slash_split"
    assert entry["parse_warning"] is None
    
    # Title then Author
    entry_rev = parser.parse_line("1. A Philosophy of Time and Space / John Doe")
    assert entry_rev["cleaned_authors"] == "John Doe"
    assert entry_rev["cleaned_title"] == "A Philosophy of Time and Space"
    assert entry_rev["orientation_decision"] == "title_then_author"
    
    # Unresolved orientation (both short)
    entry_un = parser.parse_line("2. John Doe / Jane Smith")
    assert entry_un["cleaned_authors"] is None
    assert entry_un["cleaned_title"] is None
    assert entry_un["left_part_raw"] == "John Doe"
    assert entry_un["right_part_raw"] == "Jane Smith"
    assert entry_un["orientation_decision"] == "unresolved"
    assert "Ambiguous" in entry_un["parse_warning"]
    
    # Multiple slashes
    entry2 = parser.parse_line("2. John Doe / Philosophy / Time and Space")
    assert entry2["orientation_decision"] == "author_then_title"
    assert "Multiple '/'" in entry2["parse_warning"]
    
    # No slash
    entry3 = parser.parse_line("3. John Doe - Philosophy of Time")
    assert entry3["cleaned_authors"] is None
    assert entry3["cleaned_title"] == "John Doe - Philosophy of Time"
    assert entry3["orientation_decision"] == "unresolved"
    assert entry3["parse_method"] == "raw_fallback"
    assert "No '/'" in entry3["parse_warning"]

def test_hebrew_english_mixed():
    parser = ArticleCatalogParser()
    entry = parser.parse_line("5. הרמבם / מורה נבוכים להולכים בדרך (**)")
    assert entry["article_number"] == "5"
    assert entry["cleaned_authors"] == "הרמבם"
    assert entry["cleaned_title"] == "מורה נבוכים להולכים בדרך"
    assert entry["orientation_decision"] == "author_then_title"
    assert entry["importance"] == "even_more_important"
    
    entry_title_author = parser.parse_line("6. מידה טובה ותבונה פילוסופית עמוקה / ג'ון מקדוול")
    assert entry_title_author["cleaned_authors"] == "ג'ון מקדוול"
    assert entry_title_author["cleaned_title"] == "מידה טובה ותבונה פילוסופית עמוקה"
    assert entry_title_author["orientation_decision"] == "title_then_author"
