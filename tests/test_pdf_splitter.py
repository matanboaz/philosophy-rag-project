import sys
import os
import pytest
import fitz

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.ingestion.pdf_splitter import PDFSplitter

@pytest.fixture
def mock_pdf(tmp_path):
    # Create a simple 5-page PDF for testing
    pdf_path = tmp_path / "test_source.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((50, 50), f"This is page {i+1}")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)

def test_pdf_splitter(mock_pdf, tmp_path):
    out_dir = tmp_path / "output"
    splitter = PDFSplitter(mock_pdf, str(out_dir))
    
    proposals = [
        {
            "article_number": "1",
            "cleaned_title": "First Valid Article",
            "proposed_start_page": 1,
            "proposed_end_page": 2,
            "warning": None
        },
        {
            "article_number": "2",
            "cleaned_title": "Missing Pages",
            "proposed_start_page": None,
            "proposed_end_page": None,
            "warning": "No match found"
        },
        {
            "article_number": "3",
            "cleaned_title": "Inverted Pages",
            "proposed_start_page": 4,
            "proposed_end_page": 3,
            "warning": None
        },
        {
            "article_number": "4",
            "cleaned_title": "Special! Characters?",
            "proposed_start_page": 5,
            "proposed_end_page": 5,
            "warning": None
        },
        {
            "article_number": "1",
            "cleaned_title": "First Valid Article",
            "proposed_start_page": 2,
            "proposed_end_page": 2,
            "warning": None
        },
        {
            "article_number": "5",
            "cleaned_title": "Out of Bounds",
            "proposed_start_page": 4,
            "proposed_end_page": 10,
            "warning": None
        }
    ]
    
    manifest, summary = splitter.split_and_save(proposals)
    
    # Check Summary
    assert summary["total_catalog_entries"] == 6
    assert summary["total_successfully_split"] == 4
    assert summary["total_skipped"] == 2
    assert summary["skipped_reasons"]["skipped_unmatched"] == 1
    assert summary["skipped_reasons"]["skipped_invalid_range"] == 1
    
    # Check Manifest Entries
    # Entry 1: Success
    assert manifest[0]["split_status"] == "success"
    assert manifest[0]["output_pdf_filename"] == "art_1_First_Valid_Article.pdf"
    assert os.path.exists(os.path.join(out_dir, manifest[0]["output_pdf_filename"]))
    assert manifest[0]["actual_start_page"] == 1
    assert manifest[0]["actual_end_page"] == 2
    assert manifest[0]["range_adjusted"] is False
    
    # Entry 2: Missing Pages
    assert manifest[1]["split_status"] == "skipped_unmatched"
    assert manifest[1]["output_pdf_filename"] is None
    
    # Entry 3: Inverted Pages
    assert manifest[2]["split_status"] == "skipped_invalid_range"
    assert manifest[2]["output_pdf_filename"] is None
    
    # Entry 4: Special Characters -> filename cleanup
    assert manifest[3]["split_status"] == "success"
    assert manifest[3]["output_pdf_filename"] == "art_4_Special_Characters.pdf"
    
    # Entry 5: Duplicate Filename Collision
    assert manifest[4]["split_status"] == "success"
    assert manifest[4]["output_pdf_filename"] == "art_1_First_Valid_Article_dup2.pdf"
    assert os.path.exists(os.path.join(out_dir, manifest[4]["output_pdf_filename"]))
    assert "Filename collision resolved" in manifest[4]["warning"]
    
    # Entry 6: Out of Bounds Clipping
    assert manifest[5]["split_status"] == "success_range_adjusted"
    assert manifest[5]["actual_start_page"] == 4
    assert manifest[5]["actual_end_page"] == 5 # Clipped to len(doc)
    assert manifest[5]["range_adjusted"] is True
    assert "Proposed range exceeded bounds" in manifest[5]["warning"]
    
    # Verify manifest JSON exists
    assert os.path.exists(os.path.join(out_dir, "articles_manifest.json"))
