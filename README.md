# Philosophy RAG Project

## 📚 Documentation Index
Welcome to the project documentation. Please select the guide that fits your needs:
*   [**QUICKSTART.md**](QUICKSTART.md) — For fast operator onboarding and minimal step-by-step setup.
*   [**USER_GUIDE.md**](USER_GUIDE.md) — The full, comprehensive app usage guide with UI screenshots and troubleshooting.
*   [**README.md**](#llm-backend-strategy--configuration) (This File) — The technical runbook, architecture handoff, and runtime prerequisites reference.

---

A modular, RAG-based question-answering system for a multilingual philosophy PDF corpus. The system supports document ingestion, manual article-boundary review, hybrid retrieval (vector and lexical), and grounded Hebrew answer generation with strict word budgets and guideline precedence.

## LLM Backend Strategy & Configuration
The generation pipeline abstracts the underlying LLM provider, utilizing a primary local Hebrew model with a hosted fallback.

**Supported Backends:**
1. **DICTA (Primary):** The default local Hebrew-first backend (`dicta-il/dictalm2.0-instruct`). This executes natively via HuggingFace `transformers`.
2. **Gemini (Fallback):** The hosted backend (`gemini-1.5-pro`). This executes via the official `google-genai` SDK and utilizes native structured JSON output constraints.

### Config Flags
You can route the backend execution by setting the `LLM_BACKEND_STRATEGY` environment variable:
- `$env:LLM_BACKEND_STRATEGY="dicta"` (Default): Forces the DICTA local backend.
- `$env:LLM_BACKEND_STRATEGY="gemini"`: Forces the Gemini hosted backend.
- `$env:LLM_BACKEND_STRATEGY="auto"`: Attempts to instantiate DICTA first. If local dependencies (Torch/Transformers) are missing or the model fails to load, it automatically falls back to Gemini to preserve the pipeline.

---

## ⚠️ OPERATOR NOTE: Real-World Validation Prerequisites

The pipeline architecture, prompt precedence, budget counting, and citation logic have been fully validated using internal mock stubs within an air-gapped sandbox. 

**However, a true live validation must be run on a machine that has the required runtime and network access.** 

To execute the real end-to-end generation, the operator must provide the following in their local environment:

### For Gemini Validation:
You must provide a valid API key with outbound network access to Google's servers.
```powershell
$env:GEMINI_API_KEY="your_real_api_key_here"
$env:LLM_BACKEND_STRATEGY="gemini"
py src\qa_inspector_cli.py
```

### For DICTA Validation:
You must run this on a machine with a dedicated GPU (e.g., CUDA), adequate VRAM (14GB+), and an active internet connection to download the weights from the HuggingFace Hub, alongside an installed PyTorch environment.
*Note: In `src/generation/llm_client.py` (line 21), ensure `self.pipeline` is not forcefully set to `None` so the weights download properly.*
```powershell
$env:LLM_BACKEND_STRATEGY="dicta"
py src\qa_inspector_cli.py
```
