# Philosophy RAG System: Quickstart

For full instructions and troubleshooting, see the [USER_GUIDE.md](USER_GUIDE.md).

## What this app does
*   Searches large bilingual (Hebrew/English) philosophy PDFs.
*   Answers Hebrew questions using **only** the uploaded text.
*   Provides exact page citations for every generated sentence.
*   Enforces strict word limits on answers.

## Before you start
*   Have your **Main Philosophy PDF** ready.
*   *(Optional)* Have a `.txt` file containing the titles of the articles in the PDF.
*   **AI Access:** You must have a Google Gemini API Key, OR a powerful local computer with a dedicated GPU (to run the DICTA local model).

## How to launch the app
Open your terminal in this project folder and run:
```powershell
py -m streamlit run src/app.py
```

![App Landing](docs/images/01_main_layout.png)
*Use the left sidebar to configure your AI before starting.*

---

## Fastest Normal Workflow
1.  **Launch the app** and look at the left sidebar.
2.  **Choose Backend:** Select `gemini` (and paste your API key) or `dicta` (if you have a GPU).
3.  **Upload:** In Tab 1, upload your Main PDF and Titles file. Click **Detect Boundaries**.
4.  **Approve:** Review the detected articles table and click **Approve & Split Corpus**.
5.  **Ask:** Go to Tab 2 ("Normal QA Mode") and type your question in Hebrew.
6.  **Constrain:** *(Optional)* Enter specific instructions or set a Word Budget.
7.  **Execute:** Click **Execute Query**.
8.  **Verify:** Read the answer and click **"Show Retrieved Evidence & Citations"** to check the AI's math.
9.  **Export:** Click **Download JSON** to save the session.

---

## Comparison Workflow
To compare a completely new text against your existing database:
1.  Go to **Tab 3: Batch / Comparison Mode**.
2.  Upload the new, unseen article PDF.
3.  Execute a query to automatically generate a structured report of Agreements, Contradictions, and Differences.

---

## Common Problems
*   **Red "Missing API Key" Error:** You selected `gemini` but forgot to paste your key in the sidebar.
*   **Yellow "Weak Evidence" Warning:** The AI could not find the answer in the text. It will refuse to guess.
*   **Yellow "Budget Failed" Warning:** You asked for a philosophical answer in too few words (e.g., 5 words). The AI gave up trying to compress it.
*   **"Index not found" Error:** You forgot to click **Approve & Split Corpus** in Tab 1 before asking a question.
*   **App is frozen or crashes:** You selected `dicta` but your computer lacks the heavy GPU required to run it locally. Switch to `gemini`.
