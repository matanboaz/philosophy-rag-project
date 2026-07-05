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
        schema_desc = '{"answers": [{"sub_question": "string", "answer": "string"}]}'
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
