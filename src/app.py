import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime

# Import backend modules
import sys
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.ingestion.segmenter import BoundarySegmenter
from src.ingestion.pdf_splitter import PDFSplitter
from src.retrieval.search import HybridSearcher
from src.generation.qa_pipeline import QAPipeline

# --- Configuration & State Setup ---
st.set_page_config(page_title="מערכת RAG לפילוסופיה", layout="wide")

# Initialize Session State
if "searcher" not in st.session_state:
    st.session_state.searcher = None
if "boundaries" not in st.session_state:
    st.session_state.boundaries = []
if "main_pdf_path" not in st.session_state:
    st.session_state.main_pdf_path = None
if "unseen_pdf_path" not in st.session_state:
    st.session_state.unseen_pdf_path = None

def invalidate_indices():
    st.session_state.searcher = None
    st.success("מערכת: האינדקסים בוטלו וייבנו מחדש בהפעלה הבאה.")

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
st.title("📚 מערכת RAG לפילוסופיה")

tab1, tab2, tab3 = st.tabs(["1. העלאה וגבולות (Ingestion)", "2. שאלות ותשובות (QA)", "3. אצווה / השוואה"])

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
                    if "answers" in ans_data:
                        for a in ans_data["answers"]:
                            st.markdown(f"**{a.get('sub_question', 'Q')}:** {a.get('answer', '')}")
                    elif "error" in ans_data:
                        st.error(f"שגיאת יצירה: {ans_data['error']}")
                        
                    # Render Evidence Expandable
                    with st.expander("הצג ראיות ומקורות"):
                        for c in chunks:
                            st.markdown(f"**[{c['article_title']}, p. {c['global_page_num']}]**")
                            st.text(c['text'])
                            
                    st.download_button("הורד כקובץ JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="qa_result.json")
                except Exception as e:
                    st.error(f"שגיאת ביצוע: {str(e)}")

# --- TAB 3: Batch / Comparison ---
with tab3:
    st.header("עיבוד אצווה והשוואת מאמרים חדשים")
    st.info("גרסה 1 תומכת בעיבוד אצווה ידני סדרתי.")
    
    batch_file = st.file_uploader("העלאת קובץ אצווה (JSON)", type=["json"])
    if batch_file:
        batch_data = json.load(batch_file)
        st.dataframe(pd.DataFrame(batch_data.get("queries", [])))
        if st.button("הפעל אצווה"):
            st.warning("עיבוד אצווה יופעל כאן וייצור קובץ JSON משולב.")
            # ... (Implementation of sequential loop mirroring Tab 2) ...
