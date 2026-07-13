import sys
import os
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.ingestion.segmenter import BoundaryProposer

def test_boundary_proposal():
    catalog_entries = [
        {
            "article_number": "1",
            "raw_line": "1. Author A / First Title Is A Long String",
            "cleaned_title": "First Title Is A Long String",
            "cleaned_authors": "Author A",
            "left_part_raw": "Author A",
            "right_part_raw": "First Title Is A Long String",
            "orientation_decision": "author_then_title",
            "importance": "regular"
        },
        {
            "article_number": "2",
            "raw_line": "2. Second Title Is Long / Author B",
            "cleaned_title": "Second Title Is Long",
            "cleaned_authors": "Author B",
            "left_part_raw": "Second Title Is Long",
            "right_part_raw": "Author B",
            "orientation_decision": "title_then_author",
            "importance": "more_important"
        },
        {
            "article_number": "3",
            "raw_line": "3. Short One / Short Two",
            "cleaned_title": None,
            "cleaned_authors": None,
            "left_part_raw": "Short One",
            "right_part_raw": "Short Two",
            "orientation_decision": "unresolved",
            "importance": "regular"
        }
    ]
    
    pages_text = {
        1: "This is some intro text. Not an article.",
        2: "Here begins First Title Is A Long String by Author A. This is the first article.",
        3: "First article continues here.",
        4: "And now we start Second Title Is Long written by Author B.",
        5: "Second article page 2.",
        6: "Now we have Short One. Also Short Two.",
        7: "Unresolved article page."
    }
    
    proposer = BoundaryProposer(catalog_entries)
    proposals = proposer.propose_boundaries(pages_text)
    
    assert len(proposals) == 3
    
    # Check First Article (Author / Title)
    assert proposals[0]["proposed_start_page"] == 2
    assert proposals[0]["proposed_end_page"] == 3
    assert proposals[0]["matching_method"] == "exact_title_and_author"
    
    # Check Second Article (Title / Author)
    assert proposals[1]["proposed_start_page"] == 4
    assert proposals[1]["proposed_end_page"] == 5
    assert proposals[1]["matching_method"] == "exact_title_and_author"
    
    # Check Unresolved Article (Dual Match)
    assert proposals[2]["proposed_start_page"] == 6
    assert proposals[2]["proposed_end_page"] == 7
    assert proposals[2]["matching_method"] == "ambiguous_dual_match"
