import string
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

    def _hard_cap_answer(self, final_answer, budget):
        if not budget or not isinstance(final_answer, dict) or "answers" not in final_answer: 
            return final_answer
        for ans in final_answer["answers"]:
            text = ans.get("answer", "")
            words = text.split()
            if len(words) > budget:
                ans["answer"] = " ".join(words[:budget]) + "..."
        return final_answer

    def execute_qa(self, query, chunks, guidelines=None, budget=None):
        if guidelines is None: guidelines = {}
        
        # 1. Resolve Guidelines
        applied_map, guidelines_str = self.prompts.resolve_guidelines(
            corpus_g=guidelines.get("corpus"),
            batch_g=guidelines.get("batch"),
            question_g=guidelines.get("question")
        )
        
        # 2. Build 1st Pass Prompt
        prompt = self.prompts.build_qa_prompt(query, chunks, guidelines_str, budget)
        
        # 3. Generate 1st Pass
        schema_desc = '{"answers": [{"sub_question": "string", "answer": "string", "citations": [{"chunk_id": "string", "snippet": "string"}]}]}'
        response_json = self.llm.generate_json(prompt, schema_description=schema_desc)
        
        warnings = []
        budget_failed = False
        final_answer = response_json
        
        # 4. Word Budget Enforcement (Two-Pass)
        if budget and "answers" in response_json:
            total_words = sum(self._count_hebrew_words(ans.get("answer", "")) for ans in response_json["answers"])
            upper_bound = budget * 1.10
            
            if total_words > upper_bound:
                # Trigger 2nd Pass Regeneration
                reduction_prompt = self.prompts.build_reduction_prompt(prompt, str(response_json), budget, total_words)
                response_json_2nd = self.llm.generate_json(reduction_prompt)
                
                total_words_2nd = sum(self._count_hebrew_words(ans.get("answer", "")) for ans in response_json_2nd.get("answers", []))
                
                if total_words_2nd > upper_bound:
                    budget_failed = True
                    warnings.append(f"budget_failed: 2nd pass failed to hit target {budget}. Actual: {total_words_2nd}")
                    final_answer = response_json_2nd # Fail closed
                else:
                    final_answer = response_json_2nd
            else:
                final_answer = response_json
                
        # 5. Check Weak Evidence
        ans_str = str(final_answer)
        if "אין מספיק מידע בטקסט" in ans_str:
            warnings.append("weak_evidence: true")
            
        # 6. Hard-Cap Answer Length
        final_answer = self._hard_cap_answer(final_answer, budget)
            
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

    def execute_comparison_qa(self, query, primary_chunks, reference_chunks, guidelines=None, budget=None, mode="compare"):
        if guidelines is None: guidelines = {}
        applied_map, guidelines_str = self.prompts.resolve_guidelines(
            corpus_g=guidelines.get("corpus"), batch_g=guidelines.get("batch"), question_g=guidelines.get("question")
        )
        if mode == "combine":
            prompt = self.prompts.build_combined_prompt(query, primary_chunks, reference_chunks, guidelines_str, budget)
        else:
            prompt = self.prompts.build_comparison_prompt(query, primary_chunks, reference_chunks, guidelines_str, budget)
        
        schema_desc = '{"answers": [{"sub_question": "string", "answer": "string", "citations": [{"chunk_id": "string", "snippet": "string"}]}]}'
        response_json = self.llm.generate_json(prompt, schema_description=schema_desc)
        
        warnings = []
        ans_str = str(response_json)
        if "אין מספיק מידע בטקסט" in ans_str or "weak" in ans_str.lower() or "missing" in ans_str.lower():
            warnings.append("weak_evidence: comparison missing or weak")
            
        # Hard-Cap Answer Length
        response_json = self._hard_cap_answer(response_json, budget)
            
        return {
            "query": query,
            "backend_metadata": self.llm.get_metadata(),
            "retrieved_primary": [c["chunk_id"] for c in primary_chunks],
            "retrieved_reference": [c["chunk_id"] for c in reference_chunks],
            "applied_guidelines_map": applied_map,
            "word_budget": budget,
            "raw_llm_prompt": prompt,
            "final_parsed_answer": response_json,
            "warnings": warnings
        }

    def run_comparison_batch(self, batch_json, primary_searcher, reference_searcher, global_guidelines, global_budget, mode="compare", top_k=3):
        results = []
        queries = batch_json.get("queries", [])
        
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
            primary_chunks = primary_searcher.search(query_text, top_k=top_k) if primary_searcher else []
            reference_chunks = reference_searcher.search(query_text, top_k=top_k) if reference_searcher else []
            
            # Execute
            res = self.execute_comparison_qa(
                query=query_text,
                primary_chunks=primary_chunks,
                reference_chunks=reference_chunks,
                guidelines=q_guidelines,
                budget=q_budget,
                mode=mode
            )
            
            # Pack evidence chunks directly into result for batch output
            res["evidence_chunks_primary"] = primary_chunks
            res["evidence_chunks_reference"] = reference_chunks
            
            results.append(res)
            
        return {"batch_results": results}


