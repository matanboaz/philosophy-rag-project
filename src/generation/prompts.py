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
        context_str = "\n".join([f"[CHUNK_ID: {c['chunk_id']} | SOURCE: {c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in chunks])
        
        budget_str = f"\nYou MUST limit your answer to exactly {word_budget} words." if word_budget else ""
        
        prompt = f"""
{guidelines_str}

### EVIDENCE CONTEXT ###
{context_str}

### INSTRUCTIONS ###
Answer the following query using ONLY the evidence context provided above.
If the context does not contain the answer, explicitly state exactly: "אין מספיק מידע בטקסט". DO NOT guess, DO NOT invent facts, and DO NOT use outside knowledge.
You MUST cite your sources using the structured 'citations' JSON array, providing the exact 'chunk_id' and the 'snippet' of text you relied on.
Additionally, you MUST place inline markers in your answer string using the format [CHUNK_ID] wherever that citation applies.
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

    def build_comparison_prompt(self, query, primary_chunks, reference_chunks, guidelines_str, word_budget=None):
        primary_context = "\n".join([f"[CHUNK_ID: {c['chunk_id']} | SOURCE: {c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in primary_chunks])
        ref_context = "\n".join([f"[CHUNK_ID: {c['chunk_id']} | SOURCE: {c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in reference_chunks])
        
        budget_str = f"\nYou MUST limit your answer to exactly {word_budget} words." if word_budget else ""
        
        prompt = f"""
{guidelines_str}

### NEW ARTICLE CONTEXT (PRIMARY) ###
{primary_context if primary_chunks else "No evidence found in the new article."}

### ORIGINAL CORPUS CONTEXT (REFERENCE) ###
{ref_context if reference_chunks else "No evidence found in the original corpus."}

### INSTRUCTIONS ###
You are comparing a newly uploaded article against the original reference corpus.
1. Answer the query based ONLY on the provided NEW ARTICLE CONTEXT and ORIGINAL CORPUS CONTEXT.
2. Contrast or support the answer using both contexts.
3. If the contexts lack relevant information to answer or compare, explicitly state exactly what is missing. DO NOT guess, DO NOT invent facts, and DO NOT use outside knowledge.
4. You MUST cite your sources using the structured 'citations' JSON array (with 'chunk_id' and 'snippet').
5. You MUST also place inline markers in your answer string using the format [CHUNK_ID] wherever that citation applies.
{budget_str}

### QUERY ###
{query}
"""
        return prompt

    def build_combined_prompt(self, query, primary_chunks, reference_chunks, guidelines_str, word_budget=None):
        primary_context = "\n".join([f"[CHUNK_ID: {c['chunk_id']} | SOURCE: {c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in primary_chunks])
        ref_context = "\n".join([f"[CHUNK_ID: {c['chunk_id']} | SOURCE: {c['article_title']}, p. {c['global_page_num']}]\n{c['text']}" for c in reference_chunks])
        
        budget_str = f"\nYou MUST limit your answer to exactly {word_budget} words." if word_budget else ""
        
        prompt = f"""
{guidelines_str}

### NEW ARTICLE CONTEXT (PRIMARY) ###
{primary_context if primary_chunks else "No evidence found in the new article."}

### ORIGINAL CORPUS CONTEXT (REFERENCE) ###
{ref_context if reference_chunks else "No evidence found in the original corpus."}

### INSTRUCTIONS ###
You are answering a query using both a new article and the original reference corpus.
You MUST present your final answer in two clearly labeled sections:
1. "תשובה מתוך המאמר החדש:" (Results from the selected article)
2. "תשובה מתוך מאגר הרקע:" (Results from the original background corpus)
Answer the query for each section independently using ONLY the respective context. DO NOT guess, DO NOT invent facts, and DO NOT use outside knowledge.
You MUST cite your sources using the structured 'citations' JSON array (with 'chunk_id' and 'snippet').
You MUST also place inline markers in your answer string using the format [CHUNK_ID] wherever that citation applies.
{budget_str}

### QUERY ###
{query}
"""
        return prompt
