import string
import re
from .prompts import PromptEngine
from .llm_client import LLMClientFactory

class QAPipeline:
    def __init__(self, backend_strategy=None):
        self.prompts = PromptEngine()
        self.llm = LLMClientFactory.get_client(backend_strategy)
        
    def _count_hebrew_words(self, text):
        # Strips punctuation and splits by space
        clean_text = text.translate(str.maketrans('', '', string.punctuation))
        return len([w for w in clean_text.split() if w.strip()])

    def _parse_word_constraints(self, guidelines_str, budget):
        min_words = 0
        max_words = budget if budget and budget > 0 else 999999
        
        if not guidelines_str:
            return min_words, max_words
            
        # Parse Min
        min_match = re.search(r'(?:at least|minimum|min|no less than|>|>=|לפחות|מינימום|לא פחות מ|מעל)\s*(\d+)\s*(?:words?|מיל)', guidelines_str, re.IGNORECASE)
        if min_match: 
            min_words = int(min_match.group(1))
            
        # Parse Max
        max_match = re.search(r'(?:do not exceed|maximum|max|under|no more than|<|<=|up to|לכל היותר|מקסימום|עד|לא יותר מ|מתחת ל)\s*(\d+)\s*(?:words?|מיל)', guidelines_str, re.IGNORECASE)
        if max_match: 
            parsed_max = int(max_match.group(1))
            max_words = min(max_words, parsed_max) if max_words != 999999 else parsed_max
            
        return min_words, max_words

    def _enforce_word_constraints(self, prompt, initial_json, schema_desc, min_words, max_words, status_callback=None):
        current_json = initial_json
        warnings = []
        max_retries = 2
        budget_failed = False
        
        if min_words == 0 and max_words == 999999:
            return current_json, warnings
            
        print(f"[INFO] Enforcing length constraints: min={min_words}, max={max_words}")
        
        for attempt in range(max_retries + 1):
            if "error" in current_json:
                break
                
            total_words = sum(self._count_hebrew_words(ans.get("answer", "")) for ans in current_json.get("answers", []))
            
            if status_callback:
                status_callback("Validating word count", f"Checking constraint {min_words}-{max_words} (actual: {total_words})", 70)
                
            is_too_long = total_words > (max_words * 1.10)
            is_too_short = total_words < min_words
            
            print(f"[INFO] Validation Attempt {attempt}: actual={total_words} words. Passed: {not (is_too_long or is_too_short)}")
            
            if not is_too_long and not is_too_short:
                break
                
            if attempt == max_retries:
                warnings.append(f"word_count_failed: Generated {total_words} words. Target was {min_words}-{max_words}. Retries exhausted.")
                budget_failed = True
                break
                
            warnings.append(f"Retry {attempt+1}: Generated {total_words} words. Target: {min_words}-{max_words}.")
            
            if status_callback:
                status_callback(f"Retrying generation (Attempt {attempt+1}/{max_retries})", f"Word count out of bounds ({total_words} words).", 75)
            
            retry_instruction = f"Your previous answer had {total_words} words. You MUST rewrite it to be between {min_words} and {max_words} words."
            if is_too_short:
                retry_instruction += f" Expand your analysis significantly to reach at least {min_words} words."
            if is_too_long:
                retry_instruction += f" Be more concise. Do not exceed {max_words} words."
                
            retry_prompt = f"{prompt}\n\n[STRICT SYSTEM REQUIREMENT: {retry_instruction}]"
            current_json = self.llm.generate_json(retry_prompt, schema_description=schema_desc, status_callback=status_callback)
            
        if budget_failed and "error" not in current_json:
            current_json = {"error": f"Word-count constraint could not be satisfied. Output was {total_words} words, required {min_words}-{max_words}."}
            
        return current_json, warnings

    def execute_qa(self, query, chunks, guidelines=None, budget=None, status_callback=None):
        if guidelines is None: guidelines = {}
        
        if status_callback:
            status_callback("Building prompt", "Resolving UI guidelines and chunk contexts...", 40)
            
        applied_map, guidelines_str = self.prompts.resolve_guidelines(
            corpus_g=guidelines.get("corpus"),
            batch_g=guidelines.get("batch"),
            question_g=guidelines.get("question")
        )
        
        prompt = self.prompts.build_qa_prompt(query, chunks, guidelines_str, budget)
        schema_desc = '{"answers": [{"sub_question": "string", "answer": "string", "citations": [{"chunk_id": "string", "snippet": "string"}]}]}'
        response_json = self.llm.generate_json(prompt, schema_description=schema_desc, status_callback=status_callback)
        
        min_words, max_words = self._parse_word_constraints(guidelines_str, budget)
        final_answer, length_warnings = self._enforce_word_constraints(prompt, response_json, schema_desc, min_words, max_words, status_callback)
        
        warnings = length_warnings
        ans_str = str(final_answer)
        if "אין מספיק מידע בטקסט" in ans_str:
            warnings.append("weak_evidence: true")
            
        return {
            "query": query,
            "backend_metadata": self.llm.get_metadata(),
            "retrieved_chunks": [c["chunk_id"] for c in chunks],
            "applied_guidelines_map": applied_map,
            "word_budget": budget,
            "raw_llm_prompt": prompt,
            "final_parsed_answer": final_answer,
            "warnings": warnings
        }

    def execute_comparison_qa(self, query, primary_chunks, reference_chunks, guidelines=None, budget=None, mode="compare", status_callback=None):
        if guidelines is None: guidelines = {}
        
        if status_callback:
            status_callback("Building prompt", f"Formatting multi-document context ({mode})...", 40)
            
        applied_map, guidelines_str = self.prompts.resolve_guidelines(
            corpus_g=guidelines.get("corpus"), batch_g=guidelines.get("batch"), question_g=guidelines.get("question")
        )
        if mode == "combine":
            prompt = self.prompts.build_combined_prompt(query, primary_chunks, reference_chunks, guidelines_str, budget)
        else:
            prompt = self.prompts.build_comparison_prompt(query, primary_chunks, reference_chunks, guidelines_str, budget)
        
        schema_desc = '{"answers": [{"sub_question": "string", "answer": "string", "citations": [{"chunk_id": "string", "snippet": "string"}]}]}'
        response_json = self.llm.generate_json(prompt, schema_description=schema_desc, status_callback=status_callback)
        
        min_words, max_words = self._parse_word_constraints(guidelines_str, budget)
        final_answer, length_warnings = self._enforce_word_constraints(prompt, response_json, schema_desc, min_words, max_words, status_callback)
        
        warnings = length_warnings
        ans_str = str(final_answer)
        if "אין מספיק מידע בטקסט" in ans_str or "weak" in ans_str.lower() or "missing" in ans_str.lower():
            warnings.append("weak_evidence: comparison missing or weak")
            
        return {
            "query": query,
            "backend_metadata": self.llm.get_metadata(),
            "retrieved_primary": [c["chunk_id"] for c in primary_chunks],
            "retrieved_reference": [c["chunk_id"] for c in reference_chunks],
            "applied_guidelines_map": applied_map,
            "word_budget": budget,
            "raw_llm_prompt": prompt,
            "final_parsed_answer": final_answer,
            "warnings": warnings
        }

    def run_comparison_batch(self, batch_json, primary_searcher, reference_searcher, global_guidelines, global_budget, mode="compare", top_k=3, status_callback=None):
        results = []
        queries = batch_json.get("queries", [])
        total_q = len(queries)
        
        if status_callback:
            status_callback("Initializing request", f"Starting batch processing of {total_q} questions...", 5)
        
        for q_obj in queries:
            query_text = q_obj.get("query", "")
            if not query_text:
                continue
                
            q_budget = q_obj.get("budget", global_budget)
            
            # Combine guidelines
            q_guidelines = dict(global_guidelines)
            if "guideline" in q_obj:
                q_guidelines["question"] = q_obj["guideline"]
                
            # Retrieval
            if status_callback:
                progress = int(5 + (i / total_q) * 90)
                status_callback(f"Retrieving chunks (Q{i+1}/{total_q})", f"Searching for evidence...", progress)
                
            primary_chunks = primary_searcher.search(query_text, top_k=top_k, status_callback=None) if primary_searcher else []
            reference_chunks = reference_searcher.search(query_text, top_k=top_k, status_callback=None) if reference_searcher else []
            
            # Execute
            res = self.execute_comparison_qa(
                query=query_text,
                primary_chunks=primary_chunks,
                reference_chunks=reference_chunks,
                guidelines=q_guidelines,
                budget=q_budget,
                mode=mode,
                status_callback=None # Suppress inner logs for batch mode to prevent UI flicker
            )
            
            # Pack evidence chunks directly into result for batch output
            res["evidence_chunks_primary"] = primary_chunks
            res["evidence_chunks_reference"] = reference_chunks
            
            results.append(res)
            
        return {"batch_results": results}


