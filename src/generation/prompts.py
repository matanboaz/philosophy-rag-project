import json

class PromptEngine:
    def __init__(self):
        self.system_default = "Answer strictly in Hebrew. Be highly analytical. Use only the provided context."
        
    def resolve_guidelines(self, corpus_g=None, batch_g=None, question_g=None):
        """
        Resolves guideline precedence. Lower precedence are retained but overriden explicitly if conflicting.
        Returns the applied map and the final instruction string.
        """
        applied_map = {
            "System Level": self.system_default
        }
        if corpus_g: applied_map["Corpus Level"] = corpus_g
        if batch_g: applied_map["Batch Level"] = batch_g
        if question_g: applied_map["Question Level"] = question_g
        
        instruction_string = "### GUIDELINES & PRECEDENCE ###\n"
        instruction_string += "Follow all guidelines below. Instructions labeled [Question Level] strictly override [Batch Level] and [Corpus Level] instructions if they conflict.\n"
        
        for tier, rule in applied_map.items():
            instruction_string += f"[{tier}]: {rule}\n"
            
        return applied_map, instruction_string

    def build_qa_prompt(self, query, chunks, guidelines_str, word_budget=None):
        context_str = "\n".join([f"[{c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in chunks])
        
        budget_str = f"\nYou MUST limit your answer to exactly {word_budget} words." if word_budget else ""
        
        prompt = f"""
{guidelines_str}

### EVIDENCE CONTEXT ###
{context_str}

### INSTRUCTIONS ###
Answer the following query using ONLY the evidence context provided above.
If the context does not contain the answer, explicitly state exactly: "אין מספיק מידע בטקסט".
You MUST cite your sources inline.
- Single source format: [Article Title, p. X]
- Multiple chunks supporting one sentence: [Article 1, p. X; Article 2, p. Y]
- Spanning pages: [Article 1, pp. X-Y]
Do not group citations at the bottom; put them inline at the exact sub-answer sentence.
{budget_str}

### QUERY ###
{query}
"""
        return prompt

    def build_reduction_prompt(self, original_prompt, original_answer, target_words, current_words):
        return f"""
{original_prompt}

### CORRECTION REQUIRED ###
Your previous answer was {current_words} words long. This violated the strict word budget.
You MUST rewrite the answer to be exactly or under {target_words} words, while retaining citations and Hebrew language.
Previous answer for reference:
{original_answer}
"""
