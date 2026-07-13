import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import base64
import shutil
import re
import uuid
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import backend modules
import sys
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.ingestion.segmenter import BoundarySegmenter
from src.ingestion.pdf_splitter import PDFSplitter
from src.retrieval.search import HybridSearcher
from src.retrieval.chunker import ChunkRegistryBuilder
from src.generation.qa_pipeline import QAPipeline
from src.ingestion.pdf_parser import PDFParser
import time
import time

# --- HISTORY MANAGEMENT ---
HISTORY_FILE = os.path.join(base_dir, "data", "qa_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            # Explicit UTF-8 read for history
            with open(HISTORY_FILE, "r", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        except Exception as e:
            return []
    return []

def save_history(history_list):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    # Explicit UTF-8 write for history, ensuring no ASCII-only fallback
    with open(HISTORY_FILE, "w", encoding="utf-8", errors="xmlcharrefreplace") as f:
        json.dump(history_list, f, ensure_ascii=False, indent=2)

if "tab3_history" not in st.session_state:
    st.session_state.tab3_history = load_history()

def render_answer_with_citations(answer_text, citations, all_chunks):
    """Replaces [chunk_id] in the text with a markdown tooltip and returns matched chunks for iframe rendering."""
    chunk_map = {c["chunk_id"]: c for c in all_chunks}
    cite_map = {c["chunk_id"]: c for c in citations} if citations else {}
    matched_chunks = {}
    
    def replacer(match):
        cid = match.group(1)
        chunk = chunk_map.get(cid)
        cite_info = cite_map.get(cid)
        
        if not chunk:
            return match.group(0)
            
        title = chunk.get('article_title', 'Unknown')
        page = chunk.get('global_page_num', '?')
        snippet = cite_info.get('snippet', '') if cite_info else chunk.get('text', '')[:150] + '...'
        
        snippet = snippet.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Track for the viewer panel
        if cid not in matched_chunks:
            matched_chunks[cid] = chunk
            
        # Custom HTML popover linked to the viewer anchor!
        # The pointer-events: none on the preview prevents hover flicker.
        return f'''<a href="#viewer-{cid}" class="citation-marker" style="text-decoration: none;">[{title}, עמ' {page}]
<span class="citation-preview"><strong>מקור:</strong> {title}, עמ' {page}<br><hr style="margin:6px 0; border-color:#475569;">{snippet}</span></a>'''
        
    formatted_text = re.sub(r'\[(temp_[0-9]+|art_[a-zA-Z0-9_]+)\]', replacer, answer_text)
    st.markdown(formatted_text, unsafe_allow_html=True)
    return matched_chunks

def render_pdf_viewer(chunk):
    """Renders a specific page of a PDF in an in-app iframe."""
    pdf_path = chunk.get("split_pdf_path")
    page_num = chunk.get("local_page_num", 1)
    
    if not pdf_path or not os.path.exists(pdf_path):
        st.error("קובץ המקור לא נמצא.")
        return
        
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
    # Use #page=X to jump to the exact page in the native browser viewer
    iframe_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(iframe_html, unsafe_allow_html=True)

# --- Configuration & State Setup ---
st.set_page_config(page_title="100 , לא פחות!", layout="wide")

# --- RTL CSS Injection ---
# We inject a careful, scoped RTL style that targets text, inputs, and alerts,
# while explicitly avoiding full mirroring on fragile widgets like dataframes.
RTL_STYLE = """
<style>
/* Main text, markdown, headings */
h1, h2, h3, h4, h5, h6, p, div[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right !important;
}

/* Sidebar container */
div[data-testid="stSidebar"] {
    direction: rtl;
}

/* Input labels */
div[data-testid="stTextInput"] label, 
div[data-testid="stSelectbox"] label, 
div[data-testid="stTextArea"] label, 
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label {
    direction: rtl !important;
    text-align: right !important;
    /* Removed display: flex and justify-content which break React Suspense dynamic module fetching */
}

/* Input fields (text box, area) */
input, textarea {
    direction: rtl;
    text-align: right;
}

/* Alerts / Banners */
div[data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
}

/* Tabs (Align text correctly in tab buttons) */
button[data-baseweb="tab"] div {
    direction: rtl;
    text-align: right;
}



/* Expanders */
div[data-testid="stExpander"] details summary {
    direction: rtl;
    text-align: right;
}

/* Protect fragile components from breaking under RTL */
div[data-testid="stFileUploader"] section {
    direction: ltr; /* The internal dropzone layout breaks if mirrored */
}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
    direction: ltr; /* Dataframes break completely if mirrored */
}

/* --- Citation Custom Hover Preview --- */
.citation-marker {
    position: relative;
    display: inline-block;
    cursor: pointer;
    background-color: #334155;
    color: #38bdf8;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    margin: 0 2px;
    font-weight: bold;
    text-decoration: none;
    direction: rtl;
}
.citation-marker:hover {
    background-color: #475569;
}
.citation-preview {
    visibility: hidden;
    width: max-content;
    max-width: 350px;
    background-color: #1e293b;
    color: #f8fafc;
    text-align: right;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 10px 14px;
    position: absolute;
    z-index: 9999;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.2s, visibility 0.2s;
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3), 0 4px 6px -4px rgb(0 0 0 / 0.3);
    font-weight: normal;
    font-size: 0.95rem;
    line-height: 1.5;
    white-space: pre-wrap;
    pointer-events: none;
}
.citation-marker:hover .citation-preview {
    visibility: visible;
    opacity: 1;
}
.citation-preview::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #1e293b transparent transparent transparent;
}
</style>
"""

# Use st.html if available (Streamlit >= 1.35) to prevent markdown parser leaking raw tags.
# Fallback to markdown wrapped in a zero-height container.
if hasattr(st, "html"):
    st.html(RTL_STYLE)
else:
    st.markdown(f"<div>{RTL_STYLE}</div>", unsafe_allow_html=True)


# Initialize Session State
if "searcher" not in st.session_state:
    st.session_state.searcher = None
if "boundaries" not in st.session_state:
    st.session_state.boundaries = []
if "main_pdf_path" not in st.session_state:
    st.session_state.main_pdf_path = None
if "unseen_pdf_path" not in st.session_state:
    st.session_state.unseen_pdf_path = None
if "startup_done" not in st.session_state:
    st.session_state.startup_done = False

# FORCE CACHE CLEAR PATCH: Auto-invalidate old collections to bypass Windows locks
if "searcher" in st.session_state and st.session_state.searcher is not None:
    if getattr(st.session_state.searcher, "collection", None) and getattr(st.session_state.searcher.collection, "name", "") != "philosophy_docs_v2":
        if hasattr(st.session_state.searcher, "close"):
            st.session_state.searcher.close()
        st.session_state.searcher = None

def invalidate_indices():
    st.session_state.searcher = None
    st.success("מערכת: האינדקסים בוטלו וייבנו מחדש בהפעלה הבאה.")

def get_loaded_articles():
    articles_dir = os.path.join(base_dir, "data", "processed", "new_articles")
    os.makedirs(articles_dir, exist_ok=True)
    return [d for d in os.listdir(articles_dir) 
            if os.path.isdir(os.path.join(articles_dir, d)) 
            and os.path.exists(os.path.join(articles_dir, d, "registry.jsonl"))]

def index_new_article(pdf_path, article_name):
    timestamp = int(time.time())
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', article_name)
    unique_name = f"{safe_name}_{timestamp}"
    article_dir = os.path.join(base_dir, "data", "processed", "new_articles", unique_name)
    os.makedirs(article_dir, exist_ok=True)
    
    registry_path = os.path.join(article_dir, "registry.jsonl")
    db_path = os.path.join(article_dir, "vector_store")
    
    parser = PDFParser(pdf_path)
    pages_text = parser.extract_text(bidi_reorder=True)
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=lambda x: len(tokenizer.encode(x)),
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    
    chunks = []
    for local_page_num, reordered_text in pages_text.items():
        page_chunks = text_splitter.split_text(reordered_text)
        
        for i, text_chunk in enumerate(page_chunks):
            if not text_chunk.strip(): continue
            
            chunk_id = f"art_new_{unique_name}_p{local_page_num}_c{i+1}"
            
            char_start = reordered_text.find(text_chunk)
            char_end = char_start + len(text_chunk) if char_start != -1 else -1
            line_start = reordered_text.count('\\n', 0, char_start) + 1 if char_start != -1 else -1
            line_end = reordered_text.count('\\n', 0, char_end) + 1 if char_end != -1 else -1
            
            chunks.append({
                "chunk_id": chunk_id,
                "article_id": f"new_{unique_name}",
                "article_title": article_name,
                "split_pdf_path": pdf_path,
                "global_page_num": local_page_num,
                "local_page_num": local_page_num,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_start,
                "line_end": line_end,
                "text": text_chunk
            })
            
    with open(registry_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    searcher = HybridSearcher(registry_path, db_path)
    return unique_name

def load_article_searcher(unique_name):
    article_dir = os.path.join(base_dir, "data", "processed", "new_articles", unique_name)
    registry_path = os.path.join(article_dir, "registry.jsonl")
    db_path = os.path.join(article_dir, "vector_store")
    return HybridSearcher(registry_path, db_path)

# --- Startup Preload Hook ---
if not st.session_state.startup_done:
    st.session_state.startup_done = True
    
    # 1. Main Corpus Preload & Validation
    manifest_path = os.path.join(base_dir, "data", "processed", "split_pdfs", "articles_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        articles = manifest.get("manifest", manifest) if isinstance(manifest, dict) else manifest
        split_dir = os.path.join(base_dir, "data", "processed", "split_pdfs")
        missing_files = []
        for art in articles:
            fname = art.get("output_pdf_filename")
            if fname and not os.path.exists(os.path.join(split_dir, fname)):
                missing_files.append(fname)
                
        if missing_files:
            st.error(f"🚨 שגיאת הפעלה: קבצי PDF מפוצלים חסרים מהמאגר: {missing_files[:3]}...")
        else:
            registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
            db_path = os.path.join(base_dir, "data", "vector_store")
            if os.path.exists(registry_path):
                st.session_state.searcher = HybridSearcher(registry_path, db_path)
                
    # 2. Unseen Article Preload
    startup_unseen = os.getenv("STARTUP_UNSEEN_ARTICLE")
    if startup_unseen:
        if not os.path.exists(startup_unseen):
            st.warning(f"⚠️ קובץ מאמר חדש לא נמצא בנתיב המוגדר: {startup_unseen}")
        else:
            article_name = os.path.basename(startup_unseen)
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', article_name)
            loaded_dirs = get_loaded_articles()
            
            # Check for existing index to prevent duplicates
            preloaded_name = None
            for d in loaded_dirs:
                if d.startswith(f"{safe_name}_"):
                    preloaded_name = d
                    break
                    
            if not preloaded_name:
                with st.spinner(f"טוען מאמר הפעלה אוטומטי: {article_name}..."):
                    preloaded_name = index_new_article(startup_unseen, article_name)
                    
            st.session_state.primary_searcher_name = preloaded_name
            st.session_state.primary_searcher = load_article_searcher(preloaded_name)

# --- Sidebar: Config & Backend ---
with st.sidebar:
    st.header("⚙️ הגדרות")
    backend_choice = st.selectbox("בחירת מודל שפה (Backend)", ["dicta", "gemini", "auto"])
    os.environ["LLM_BACKEND_STRATEGY"] = backend_choice
    
    st.subheader("מפתחות API")
    gemini_key = st.text_input("מפתח API של Gemini", type="password")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    elif "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    st.divider()
    st.markdown("### הנחיות (Guidelines)")
    corpus_guide = st.text_area("הנחיות ברמת המאגר (Corpus)", "Answer in Hebrew. Be analytical.")
    batch_guide = st.text_area("הנחיות ברמת האצווה (Batch)", "")
    
# --- Main App ---
st.title("100 , לא פחות!")

tab1, tab2, tab3 = st.tabs(["1. העלאה וגבולות", "2. שאלות ותשובות (QA)", "3. מאמר חדש / השוואה"])

# --- TAB 1: Ingestion ---
with tab1:
    st.header("העלאת מאגר")
    main_pdf = st.file_uploader("העלאת קובץ PDF ראשי", type=["pdf"], key="main_pdf")
    
    if main_pdf:
        # Save temp file
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        pdf_path = os.path.join(raw_dir, main_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(main_pdf.getbuffer())
        
        if st.session_state.main_pdf_path != pdf_path:
            st.session_state.main_pdf_path = pdf_path
            invalidate_indices()
            st.session_state.boundaries = []
            
        st.success(f"הועלה: {main_pdf.name}")
        
        title_file = st.file_uploader("העלאת קובץ כותרות טקסט (אופציונלי)", type=["txt"])
        
        if st.button("זיהוי גבולות המאמרים"):
            with st.spinner("מנתח את מבנה המסמך..."):
                titles = []
                if title_file:
                    titles = [line.decode("utf-8").strip() for line in title_file.readlines() if line.strip()]
                segmenter = BoundarySegmenter(pdf_path)
                st.session_state.boundaries = segmenter.detect_boundaries(titles)
                st.success("זוהו גבולות!")

    if st.session_state.boundaries:
        st.subheader("סקירת גבולות המאמרים")
        df = pd.DataFrame(st.session_state.boundaries)
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        
        if st.button("אישור ופיצול מאגר"):
            with st.spinner("מפצל קבצים ושומר נתוני מטא..."):
                final_boundaries = edited_df.to_dict('records')
                splitter = PDFSplitter(st.session_state.main_pdf_path)
                out_dir = os.path.join(base_dir, "data", "processed", "split_pdfs")
                splitter.split_pdf(final_boundaries, out_dir)
                
                # Automatically build global RAG index for Tab 2
                st.info("בונה אינדקס חיפוש למאגר הכללי (Tab 2)...")
                meta_path = os.path.join(out_dir, "articles_manifest.json")
                registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
                builder = ChunkRegistryBuilder(meta_path, registry_path)
                builder.build_registry()
                
                invalidate_indices() # Rebuild indices on next query
                st.success("אושר! המאמרים פוצלו והמידע נשמר.")

# --- TAB 2: Normal QA Mode ---
with tab2:
    st.header("חיפוש במאגר")
    query = st.text_input("הכנס שאלה:")
    q_guide = st.text_input("הנחיה ספציפית לשאלה (עוקפת הגדרות כלליות):")
    budget = st.number_input("תקציב מילים (0 ללא הגבלה)", min_value=0, value=15)
    
    if st.button("בצע שאילתה"):
        if backend_choice == "gemini" and not gemini_key:
            st.error("🚨 חסר מפתח API של Gemini! אנא הזן אותו בסרגל הצד.")
        else:
            with st.spinner("מאחזר ומסכם..."):
                try:
                    # Lazy load indices
                    if not st.session_state.searcher:
                        registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
                        db_path = os.path.join(base_dir, "data", "vector_store")
                        if os.path.exists(registry_path):
                            st.session_state.searcher = HybridSearcher(registry_path, db_path)
                        else:
                            st.error("האינדקס לא נמצא. אנא בצע 'אישור ופיצול מאגר' תחילה.")
                            st.stop()
                    
                    searcher = st.session_state.searcher
                    chunks = searcher.search(query, top_k=3)
                    
                    pipeline = QAPipeline(backend_strategy=backend_choice)
                    guidelines_map = {"corpus": corpus_guide, "batch": batch_guide, "question": q_guide}
                    
                    # Note: We use budget=budget only if > 0
                    result = pipeline.execute_qa(query, chunks, guidelines_map, budget if budget > 0 else None)
                    
                    # Render Warnings
                    if result["warnings"]:
                        for w in result["warnings"]:
                            if "budget_failed" in w:
                                st.warning(f"⚠️ חריגה מתקציב המילים: {w}")
                            elif "weak_evidence" in w:
                                st.warning("⚠️ ראיות חלשות: המודל דיווח שאין מספיק מידע בטקסט.")
                            else:
                                st.warning(f"⚠️ אזהרה: {w}")
                                
                    # Render Answer
                    st.markdown("### תשובה")
                    ans_data = result["final_parsed_answer"]
                    used_chunks = {}
                    if "answers" in ans_data:
                        for a in ans_data["answers"]:
                            st.markdown(f"**{a.get('sub_question', 'Q')}:**")
                            matched = render_answer_with_citations(a.get('answer', ''), a.get('citations', []), chunks)
                            used_chunks.update(matched)
                    elif "error" in ans_data:
                        st.error(f"שגיאת יצירה: {ans_data['error']}")
                        
                    # Render Document Viewer Panels
                    if used_chunks:
                        st.markdown("### 📄 מסמכי מקור (Source Viewers)")
                        for cid, chunk in used_chunks.items():
                            title = chunk.get('article_title', 'Unknown')
                            page = chunk.get('global_page_num', '?')
                            st.markdown(f'<div id="viewer-{cid}" style="position:relative; top:-80px;"></div>', unsafe_allow_html=True)
                            with st.expander(f"📖 הצג מסמך מקור: {title} (עמ' {page})"):
                                render_pdf_viewer(chunk)
                                
                    # Render Raw Evidence Expandable
                    with st.expander("הצג ראיות ומקורות"):
                        for c in chunks:
                            st.markdown(f"**[{c['article_title']}, p. {c['global_page_num']}]**")
                            st.text(c['text'])
                            
                    st.download_button("הורד כקובץ JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="qa_result.json")
                except Exception as e:
                    st.error(f"שגיאת ביצוע: {str(e)}")

# --- TAB 3: Batch / Comparison ---
with tab3:
    st.header("עיבוד מאמר חדש / השוואה")
    st.info("העלה מאמר חדש (שאינו במאגר) כדי להשתמש בו כמקור ראשי לתשובה, עם אפשרות השוואה למאגר הרקע, באצווה או בשאלה בודדת.")
    
    st.markdown("### 1. ניהול מאמרים (Article Management)")
    new_article = st.file_uploader("העלאת מאמר חדש (PDF) - מקור ראשי", type=["pdf"], key="new_article")
    
    if new_article:
        temp_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, new_article.name)
        with open(pdf_path, "wb") as f:
            f.write(new_article.getbuffer())
            
        with st.spinner("בונה אינדקס קבוע למאמר החדש..."):
            unique_name = index_new_article(pdf_path, new_article.name)
            # Force the dropdown to select this newly uploaded article
            st.session_state.primary_searcher_name = unique_name
            st.session_state.primary_searcher = load_article_searcher(unique_name)
            st.session_state.selected_primary_article = unique_name
        st.success(f"המאמר נטען ונשמר בהצלחה: {unique_name}")
        
    with st.expander("ייבוא תיקיית מאמרים מפוצלים קיימים (Import Folder)"):
        default_import_path = os.path.join(base_dir, "data", "processed", "split_pdfs")
        import_folder = st.text_input("נתיב תיקיית ה-PDF לטעינה:", value=default_import_path)
        if st.button("ייבא מאמרים מתיקייה"):
            if not os.path.exists(import_folder):
                st.error("התיקייה לא נמצאה.")
            else:
                with st.spinner("סורק ומייבא קבצים..."):
                    found = 0
                    imported = 0
                    skipped = 0
                    failed = []
                    
                    # Compute existing bases to avoid duplicates
                    bases = set()
                    for d in get_loaded_articles():
                        parts = d.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            bases.add(parts[0])
                        else:
                            bases.add(d)
                            
                    for file in os.listdir(import_folder):
                        if file.lower().endswith('.pdf'):
                            found += 1
                            if file in bases:
                                skipped += 1
                                continue
                            try:
                                pdf_path = os.path.join(import_folder, file)
                                index_new_article(pdf_path, file)
                                imported += 1
                            except Exception as e:
                                print(f"Error importing {file}: {e}")
                                failed.append(file)
                                
                    st.success(f"סריקה הושלמה! נמצאו: {found}, יובאו חדשים: {imported}, דולגו (כבר קיימים): {skipped}")
                    if failed:
                        st.error(f"נכשלו: {', '.join(failed)}")

    loaded_articles = get_loaded_articles()
    
    # --- New Source of Truth for Dropdown ---
    manifest_path = os.path.join(base_dir, "data", "processed", "split_pdfs", "articles_manifest.json")
    valid_options = []
    display_labels = {}
    
    # 1. Add valid manifest articles
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        articles = manifest.get("manifest", manifest) if isinstance(manifest, dict) else manifest
        
        for art in articles:
            fname = art.get("output_pdf_filename")
            if not fname: continue
            
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', fname)
            # Find matching indexed folder
            matching_dir = next((d for d in loaded_articles if d.startswith(f"{safe_name}_")), None)
            if matching_dir:
                valid_options.append(matching_dir)
                display_labels[matching_dir] = f"📚 {art.get('raw_line') or art.get('cleaned_title')}"
                
    # 2. Add the active unseen article (from manual upload or startup hook)
    active_unseen = st.session_state.get("primary_searcher_name")
    if active_unseen and active_unseen not in valid_options:
        valid_options.append(active_unseen)
        # Extract original filename by stripping the timestamp
        clean_unseen = active_unseen.rsplit('_', 1)[0]
        display_labels[active_unseen] = f"📄 [Unseen / New] {clean_unseen}"
        
    if valid_options:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_article = st.selectbox(
                "בחר מאמר ראשי", 
                valid_options, 
                format_func=lambda x: display_labels.get(x, x),
                key="selected_primary_article"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("מחק מאמר נבחר"):
                if "primary_searcher_name" in st.session_state and st.session_state.primary_searcher_name == selected_article:
                    if hasattr(st.session_state.primary_searcher, 'close'):
                        st.session_state.primary_searcher.close()
                    del st.session_state.primary_searcher
                    del st.session_state.primary_searcher_name
                    import gc; gc.collect()
                
                import time; time.sleep(0.5)
                
                target_dir = os.path.join(base_dir, "data", "processed", "new_articles", selected_article)
                try:
                    shutil.rmtree(target_dir)
                except PermissionError:
                    # Windows holds onto memory-mapped files from ChromaDB. 
                    # We aggressively delete everything else (like registry.jsonl) 
                    # so the app will ignore this folder from now on.
                    shutil.rmtree(target_dir, ignore_errors=True)
                    
                st.success("המאמר נמחק בהצלחה.")
                st.rerun()
                
        if selected_article:
            if "primary_searcher_name" not in st.session_state or st.session_state.primary_searcher_name != selected_article:
                st.session_state.primary_searcher = load_article_searcher(selected_article)
                st.session_state.primary_searcher_name = selected_article
                st.session_state.tab3_answer_state = None
                
            st.markdown("### 2. אסטרטגיה (Strategy)")
            comp_strategy = st.selectbox("אסטרטגיית השוואה", ["מאמר חדש בלבד", "השוואה למאגר הרקע המקורי", "גם מאמר חדש וגם מאגר הרקע המקורי"])
            
            st.markdown("### 3. ביצוע (Execution)")
            exec_mode = st.radio("מצב ביצוע", ["שאלה בודדת", "שאלות באצווה (JSON)"])
            
            if exec_mode == "שאלה בודדת":
                comp_query = st.text_input("הכנס שאלה למאמר החדש:", key="comp_query")
                comp_guide = st.text_input("הנחיה ספציפית לשאלה:", key="comp_guide")
                comp_budget = st.number_input("תקציב מילים (0 ללא הגבלה)", min_value=0, value=25, key="comp_budget")
                
                if st.button("בצע שאילתת השוואה"):
                    if backend_choice == "gemini" and not gemini_key:
                        st.error("🚨 חסר מפתח API של Gemini! אנא הזן אותו בסרגל הצד.")
                    else:
                        primary_chunks = []
                        reference_chunks = []
                        
                        if comp_strategy in ["השוואה למאגר הרקע המקורי", "גם מאמר חדש וגם מאגר הרקע המקורי"]:
                            if not st.session_state.searcher:
                                registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
                                db_path = os.path.join(base_dir, "data", "vector_store")
                                if os.path.exists(registry_path):
                                    st.session_state.searcher = HybridSearcher(registry_path, db_path)
                                else:
                                    st.error("🚨 שגיאה: מאגר הרקע המקורי לא נטען. חזור ללשונית 1 כדי להעלות ולאשר אותו, או בחר 'מאמר חדש בלבד'.")
                                    st.stop()
                                    
                        with st.spinner("מאחזר ומשווה..."):
                            try:
                                primary_chunks = st.session_state.primary_searcher.search(comp_query, top_k=3)
                                
                                # --- BUG 2 FIX: METADATA INJECTION FOR IDENTITY QUESTIONS ---
                                deterministic_ans = None
                                if primary_chunks:
                                    meta_title = primary_chunks[0].get("article_title", "Unknown")
                                    author_cue = meta_title.split('-')[0].strip() if '-' in meta_title else "לא ידוע"
                                    title_cue = meta_title.split('-')[1].strip() if '-' in meta_title else meta_title
                                    
                                    meta_chunk = {
                                        "chunk_id": "metadata_header",
                                        "article_title": meta_title,
                                        "global_page_num": "Metadata",
                                        "text": f"DOCUMENT IDENTITY METADATA:\\nFilename: {meta_title}\\nAuthor: {author_cue}\\nTitle: {title_cue}\\nCRITICAL INSTRUCTION: If asked for author or title, you MUST use these exact extracted values. DO NOT say there is insufficient information."
                                    }
                                    primary_chunks.insert(0, meta_chunk)
                                    
                                    # Short-circuit logic for pure identity questions
                                    id_keywords = ["מה שם מחבר", "מי המחבר", "מי כתב", "מה כותרת המאמר", "שם המאמר"]
                                    if any(k in comp_query for k in id_keywords) and len(comp_query.split()) <= 10:
                                        deterministic_ans = f"מחבר המאמר הוא {author_cue} ושם המאמר הוא {title_cue}."
                                        
                                if comp_strategy in ["השוואה למאגר הרקע המקורי", "גם מאמר חדש וגם מאגר הרקע המקורי"]:
                                    reference_chunks = st.session_state.searcher.search(comp_query, top_k=3)
                                    
                                pipeline = QAPipeline(backend_strategy=backend_choice)
                                guidelines_map = {"corpus": corpus_guide, "batch": batch_guide, "question": comp_guide}
                                
                                # Removed ASCII-crashing debug prints that attempted to log Hebrew strategy strings
                                
                                if deterministic_ans and comp_strategy == "מאמר חדש בלבד":
                                    result = {
                                        "warnings": [],
                                        "final_parsed_answer": {
                                            "answers": [{
                                                "sub_question": comp_query,
                                                "answer": deterministic_ans,
                                                "citations": [{"chunk_id": "metadata_header", "snippet": meta_title}]
                                            }]
                                        }
                                    }
                                elif comp_strategy == "מאמר חדש בלבד":
                                    result = pipeline.execute_qa(comp_query, primary_chunks, guidelines_map, comp_budget if comp_budget > 0 else None)
                                else:
                                    mode_flag = "combine" if comp_strategy == "גם מאמר חדש וגם מאגר הרקע המקורי" else "compare"
                                    result = pipeline.execute_comparison_qa(comp_query, primary_chunks, reference_chunks, guidelines_map, comp_budget if comp_budget > 0 else None, mode=mode_flag)
                                
                                st.session_state.tab3_answer_state = {
                                    "warnings": result.get("warnings", []),
                                    "ans_data": result.get("final_parsed_answer", {}),
                                    "primary_chunks": primary_chunks,
                                    "reference_chunks": reference_chunks,
                                    "comp_strategy": comp_strategy
                                }
                                
                                # Capture History
                                new_entry = {
                                    "id": str(uuid.uuid4()),
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "main_article": selected_article,
                                    "strategy": comp_strategy,
                                    "question": comp_query,
                                    "answer_data": result.get("final_parsed_answer", {}),
                                    "is_deterministic": bool(deterministic_ans),
                                    "primary_chunks": primary_chunks,
                                    "reference_chunks": reference_chunks
                                }
                                st.session_state.tab3_history.insert(0, new_entry)
                                save_history(st.session_state.tab3_history)
                                
                            except Exception as e:
                                st.error(f"שגיאת ביצוע: {str(e)}")
                                
                st.markdown("### 4. היסטוריית שאלות (History)")
                if st.session_state.tab3_history:
                    with st.expander("הצג היסטוריית שאלות קודמות", expanded=False):
                        if st.button("נקה היסטוריה", key="clear_hist"):
                            st.session_state.tab3_history = []
                            save_history([])
                            st.rerun()
                            
                        for i, entry in enumerate(st.session_state.tab3_history):
                            c1, c2 = st.columns([8, 1])
                            with c1:
                                lbl = f"🕒 {entry['timestamp']} | ❓ {entry['question'][:50]}"
                                if st.button(lbl, key=f"hist_btn_{entry['id']}", use_container_width=True):
                                    st.session_state.tab3_answer_state = {
                                        "warnings": [],
                                        "ans_data": entry.get("answer_data", {}),
                                        "primary_chunks": entry.get("primary_chunks", []),
                                        "reference_chunks": entry.get("reference_chunks", []),
                                        "comp_strategy": entry.get("strategy", "")
                                    }
                            with c2:
                                if st.button("❌", key=f"hist_del_{entry['id']}"):
                                    st.session_state.tab3_history.pop(i)
                                    save_history(st.session_state.tab3_history)
                                    st.rerun()
                else:
                    st.write("אין היסטוריה זמינה.")
                    
                st.markdown("---")
                                
                if st.session_state.get("tab3_answer_state"):
                    state = st.session_state.tab3_answer_state
                    if state["warnings"]:
                        for w in state["warnings"]:
                            if "weak_evidence" in w:
                                st.warning("⚠️ ראיות חלשות: חסר מידע באחד המקורות להשוואה.")
                            else:
                                st.warning(f"⚠️ {w}")
                                
                    st.markdown("### תשובה")
                    st.caption("התשובה מיוצרת בעברית על סמך ראיות משפת המקור.")
                    ans_data = state["ans_data"]
                    primary_chunks = state["primary_chunks"]
                    reference_chunks = state["reference_chunks"]
                    comp_strategy = state["comp_strategy"]
                    all_chunks = primary_chunks + reference_chunks
                    used_chunks = {}
                    
                    if "answers" in ans_data:
                        for a in ans_data["answers"]:
                            st.markdown(f"**{a.get('sub_question', 'Q')}:**")
                            matched = render_answer_with_citations(a.get('answer', ''), a.get('citations', []), all_chunks)
                            used_chunks.update(matched)
                    elif "error" in ans_data:
                        st.error(f"שגיאת יצירה: {ans_data['error']}")
                        
                    if used_chunks:
                        st.markdown("### 📄 מסמכי מקור (Source Viewers)")
                        for cid, chunk in used_chunks.items():
                            title = chunk.get('article_title', 'Unknown')
                            page = chunk.get('global_page_num', '?')
                            st.markdown(f'<div id="viewer-{cid}" style="position:relative; top:-80px;"></div>', unsafe_allow_html=True)
                            with st.expander(f"📖 הצג מסמך מקור: {title} (עמ' {page})"):
                                render_pdf_viewer(chunk)
                        
                    with st.expander("🔍 ראיות מהמאמר החדש (מוצגות בשפת המקור)"):
                        if primary_chunks:
                            for c in primary_chunks:
                                st.markdown(f"**[{c['article_title']}, p. {c['global_page_num']}]**")
                                safe_text = c['text'].replace('\\n', '<br>')
                                st.markdown(f'<div dir="rtl" style="text-align: right;">{safe_text}</div>', unsafe_allow_html=True)
                        else:
                            st.write("לא נמצאו ראיות.")
                            
                    if comp_strategy in ["השוואה למאגר הרקע המקורי", "גם מאמר חדש וגם מאגר הרקע המקורי"]:
                        with st.expander("📚 ראיות ממאגר הרקע המקורי (מוצגות בשפת המקור)"):
                            if reference_chunks:
                                for c in reference_chunks:
                                    st.markdown(f"**[{c['article_title']}, p. {c['global_page_num']}]**")
                                    safe_text = c['text'].replace('\\n', '<br>')
                                    st.markdown(f'<div dir="rtl" style="text-align: right;">{safe_text}</div>', unsafe_allow_html=True)
                            else:
                                st.write("לא נמצאו ראיות במאגר הרקע.")
            
            elif exec_mode == "שאלות באצווה (JSON)":
                batch_file = st.file_uploader("העלאת קובץ אצווה (JSON)", type=["json"], key="batch_file")
                if batch_file:
                    batch_data = json.load(batch_file)
                    st.dataframe(pd.DataFrame(batch_data.get("queries", [])))
                    
                    if st.button("הפעל אצווה"):
                        if backend_choice == "gemini" and not gemini_key:
                            st.error("🚨 חסר מפתח API של Gemini! אנא הזן אותו בסרגל הצד.")
                        else:
                            reference_searcher = None
                            if comp_strategy in ["השוואה למאגר הרקע המקורי", "גם מאמר חדש וגם מאגר הרקע המקורי"]:
                                if not st.session_state.searcher:
                                    registry_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
                                    db_path = os.path.join(base_dir, "data", "vector_store")
                                    if os.path.exists(registry_path):
                                        st.session_state.searcher = HybridSearcher(registry_path, db_path)
                                    else:
                                        st.error("🚨 שגיאה: מאגר הרקע המקורי לא נטען.")
                                        st.stop()
                                reference_searcher = st.session_state.searcher
                                
                            with st.spinner("מעבד שאלות באצווה..."):
                                try:
                                    pipeline = QAPipeline(backend_strategy=backend_choice)
                                    global_guidelines = {"corpus": corpus_guide, "batch": batch_guide}
                                    
                                    mode_flag = "combine" if comp_strategy == "גם מאמר חדש וגם מאגר הרקע המקורי" else "compare"
                                    batch_results = pipeline.run_comparison_batch(
                                        batch_json=batch_data,
                                        primary_searcher=st.session_state.primary_searcher,
                                        reference_searcher=reference_searcher,
                                        global_guidelines=global_guidelines,
                                        global_budget=25,
                                        mode=mode_flag
                                    )
                                    
                                    out_dir = os.path.join(base_dir, "data", "processed", "generation_logs")
                                    os.makedirs(out_dir, exist_ok=True)
                                    out_path = os.path.join(out_dir, f"batch_comparison_{int(time.time())}.json")
                                    
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump(batch_results, f, ensure_ascii=False, indent=2)
                                        
                                    st.success(f"אצווה הושלמה! התוצאות נשמרו ב- {out_path}")
                                    st.download_button("הורד תוצאות אצווה (JSON)", data=json.dumps(batch_results, ensure_ascii=False, indent=2), file_name="batch_results.json")
                                    
                                except Exception as e:
                                    st.error(f"שגיאת אצווה: {str(e)}")
    else:
        st.info("לא נטענו מאמרים חדשים. העלה מאמר כדי להתחיל.")
