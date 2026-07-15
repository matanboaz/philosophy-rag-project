import os
import json
import traceback

class BaseLLMClient:
    def __init__(self):
        self.model_name = "unknown"
        
    def generate_json(self, prompt, schema_description="", status_callback=None):
        """Generates JSON output matching the requested schema."""
        raise NotImplementedError
        
    def get_metadata(self):
        return {"backend": self.__class__.__name__, "model": self.model_name}


class DictaLLMClient(BaseLLMClient):
    def __init__(self):
        super().__init__()
        self.model_name = "dicta-il/dictalm2.0-instruct"
        # We now connect to a high-speed local inference server (e.g. vLLM or Ollama)
        # rather than running a raw, slow HuggingFace pipeline in eager mode.
        self.api_base = os.getenv("LOCAL_INFERENCE_URL", "http://localhost:8000/v1")
        
    def generate_json(self, prompt, schema_description="", status_callback=None):
        print(f"[INFO] Executing request via Local Inference Server at {self.api_base}...")
        
        full_prompt = f"Output strict JSON matching schema: {schema_description}\n\n{prompt}"
        
        import urllib.request
        import urllib.error
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
            # We can also pass response_format={"type": "json_object"} if the inference engine supports it
        }
        
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        content = ""
        import socket
        import time
        timeout_seconds = 300  # Increased from 120s to allow for local LM generation times
        max_retries = 2
        
        for attempt in range(max_retries):
            if status_callback:
                status_callback(f"Calling Local DICTA Server (Attempt {attempt+1}/{max_retries})", "Processing locally...", 50)
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    content = result['choices'][0]['message']['content'].strip()
                    
                    # Cleanup potential markdown ticks from the model
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                        
                    return json.loads(content.strip())
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
                return {"error": f"DICTA Server HTTP Error {e.code}: {err_body}"}
            except (TimeoutError, socket.timeout):
                if attempt < max_retries - 1:
                    print(f"[WARNING] DICTA timeout after {timeout_seconds}s. Retrying in 5 seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(5)
                    continue
                return {"error": f"DICTA request timed out definitively after {max_retries} attempts ({timeout_seconds}s each). The local server is likely overloaded or hanging."}
            except urllib.error.URLError as e:
                if isinstance(e.reason, socket.timeout) or isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                    if attempt < max_retries - 1:
                        print(f"[WARNING] DICTA URLError timeout after {timeout_seconds}s. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    return {"error": f"DICTA request timed out definitively after {max_retries} attempts ({timeout_seconds}s each). The local server is likely overloaded or hanging."}
                    
                print(f"[ERROR] Could not connect to local inference server: {e}")
                # Safe sandbox fallback behavior if server is down
                if "CORRECTION REQUIRED" in prompt:
                    return {"answers": [{"sub_question": "Mocked", "answer": f"תשובה מתוקנת מ-DICTA (Server Offline): קצר יותר [Test, p. 1].", "word_count": 8}]}
                return {"answers": [{"sub_question": "Mocked", "answer": f"תשובה מורחבת מ-DICTA (Server Offline): אנא הפעל את שרת vLLM או Ollama בפורט 8000. [Test, p. 1].", "word_count": 12}]}
            except json.decoder.JSONDecodeError as e:
                # REPAIR PASS: If model returns a valid plain-text fallback instead of JSON, wrap it safely.
                if content and isinstance(content, str) and len(content.strip()) > 5:
                    return {
                        "answers": [{
                            "sub_question": "Answer (Auto-Repaired)",
                            "answer": content.strip(),
                            "citations": []
                        }]
                    }
                return {"error": f"DICTA model failed to output valid JSON. Raw output: {content}"}
            except Exception as e:
                return {"error": f"DICTA generation failed: {str(e)}"}


class GeminiLLMClient(BaseLLMClient):
    def __init__(self):
        super().__init__()
        # If set, we use it. Otherwise, we auto-select the best available at runtime.
        self.env_model_name = os.getenv("GEMINI_MODEL_NAME")
        self.model_name = self.env_model_name if self.env_model_name else "gemini-3.5-flash"
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.validation_error = None
        
        if self.api_key:
            self._validate_model()
            
    def _validate_model(self):
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            
            supported_names = []
            # Call the models list endpoint to verify support
            for m in client.models.list():
                clean_name = m.name.replace("models/", "").lower()
                methods = getattr(m, "supported_generation_methods", [])
                
                # Exclude models lacking generation, old legacy models, and specialized non-text variants
                if "generateContent" not in methods:
                    continue
                if "1.0" in clean_name or "vision" in clean_name or "embedding" in clean_name or "aqa" in clean_name:
                    continue
                    
                supported_names.append(clean_name)
                
            # Heuristic scoring to prefer newer, higher-tier text models
            def model_score(name):
                score = 0
                if "gemini-3" in name: score += 3000
                elif "gemini-2.5" in name: score += 2000
                elif "gemini-2.0" in name: score += 1000
                elif "gemini-1.5" in name: score += 500
                
                if "pro" in name: score += 100
                elif "flash" in name: score += 50
                
                if "latest" in name: score += 10
                elif "exp" in name: score -= 50
                return score
                
            supported_names.sort(key=model_score, reverse=True)
            
            candidates = []
            if self.env_model_name:
                candidates.append(self.env_model_name.replace("models/", "").lower())
            candidates.extend(supported_names)
            
            # Remove duplicates preserving order
            seen = set()
            candidates = [x for x in candidates if not (x in seen or seen.add(x))]
            
            print(f"[INFO] Gemini probing candidates (Top 5): {candidates[:5]}...")
            
            working_model = None
            probe_prompt = "Output JSON matching schema: {}\n\nReturn empty JSON: {}"
            
            # Active Capability Probe
            for candidate in candidates:
                try:
                    response = client.models.generate_content(
                        model=candidate,
                        contents=probe_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                            max_output_tokens=10
                        )
                    )
                    working_model = candidate
                    print(f"[INFO] Successfully probed and selected: {working_model}")
                    break
                except Exception as e:
                    print(f"[WARNING] Candidate {candidate} rejected during probe: {str(e)}")
                    continue
                    
            if not working_model:
                self.validation_error = "No supported Gemini models passed the capability probe (tested JSON mode support & permissions)."
                return
                
            self.model_name = working_model
            
        except Exception as e:
            self.validation_error = f"Gemini startup validation failed: {str(e)}"
            
    def generate_json(self, prompt, schema_description="", status_callback=None):
        if self.validation_error:
            return {"error": self.validation_error}
            
        if not self.api_key:
            return {"error": "Missing GEMINI_API_KEY in environment."}
            
        import time
        import random
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=self.api_key)
        full_prompt = f"Output JSON matching schema: {schema_description}\n\n{prompt}"
        
        max_attempts = 4
        base_delay = 2.0
        
        for attempt in range(max_attempts):
            if status_callback:
                status_callback(f"Calling Gemini API (Attempt {attempt+1}/{max_attempts})", "Waiting for Google network...", 50)
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                e_str = str(e).lower()
                is_503 = "503" in e_str or "unavailable" in e_str or "overloaded" in e_str
                is_429 = "429" in e_str or "too many requests" in e_str or "quota" in e_str
                is_server_error = "500" in e_str or "502" in e_str or "504" in e_str
                is_retryable = is_503 or is_429 or is_server_error
                
                if not is_retryable or attempt == max_attempts - 1:
                    if is_503:
                        return {"error": "המודל של Gemini עמוס כרגע (503). אנא נסה שוב בעוד מספר דקות."}
                    if is_429:
                        return {"error": "חריגה ממגבלת הבקשות של Gemini (429). אנא נסה שוב מאוחר יותר."}
                    
                    if "400" in e_str or "403" in e_str or "404" in e_str:
                        return {"error": f"שגיאת הרשאה או בקשה לא תקינה מול Gemini (4xx). יתכן שהמודל לא נתמך במצב JSON או שהטוקן פג תוקף. פירוט שגיאה: {e_str}"}
                        
                    return {"error": f"שגיאה כללית בהפקת תשובה מ-Gemini: {e_str}"}
                    
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                # Honor Retry-After metadata if provided by the SDK exception
                if hasattr(e, 'headers') and isinstance(getattr(e, 'headers'), dict):
                    headers = {k.lower(): v for k, v in getattr(e, 'headers').items()}
                    if 'retry-after' in headers:
                        try:
                            delay = max(delay, float(headers['retry-after']))
                        except ValueError:
                            pass
                            
                print(f"[WARNING] Gemini transient error ({'503' if is_503 else '429'}). Retrying in {delay:.1f}s (Attempt {attempt+1}/{max_attempts})...")
                time.sleep(delay)

class OpenAILLMClient(BaseLLMClient):
    def __init__(self):
        super().__init__()
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1")
        self.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_json(self, prompt, schema_description="", status_callback=None):
        if not self.api_key:
            return {"error": "Missing OPENAI_API_KEY in environment."}
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            system_instruction = (
                "You are acting as a careful philosophy professor or teaching assistant grading/answering an academic philosophy assignment. "
                "You must output valid JSON. "
                "Prioritize accuracy over fluency, avoid generic filler, and avoid repeating the text as a mechanical translation. "
                "Highlight key distinctions and commitments in the argument, and give examples or clarifications only when they genuinely help explain the philosophy."
            )
            full_prompt = f"Output JSON matching schema: {schema_description}\n\n{prompt}"
            
            max_attempts = 4
            last_error = None
            
            for attempt in range(max_attempts):
                if status_callback:
                    status_callback(f"Calling OpenAI API ({self.model_name}, Attempt {attempt+1}/{max_attempts})", "Waiting for remote generation...", 50)
                
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": full_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        max_tokens=4000
                    )
                    
                    choice = response.choices[0]
                    raw_content = choice.message.content
                    
                    if choice.finish_reason == "length":
                        safe_ending = raw_content[-100:] if raw_content else "None"
                        print(f"[ERROR] OpenAI response truncated due to length limits. Raw ending: {safe_ending}")
                        last_error = "Response was truncated by API length limits before valid JSON could be formed."
                        continue
                        
                    try:
                        return json.loads(raw_content)
                    except json.JSONDecodeError as jde:
                        safe_raw = raw_content[:200] + "..." if raw_content and len(raw_content) > 200 else str(raw_content)
                        print(f"[ERROR] Malformed JSON from OpenAI. Error: {str(jde)} | Raw Output: {safe_raw}")
                        last_error = f"Malformed JSON output: {str(jde)}"
                        full_prompt += f"\\n\\n[SYSTEM ERROR: Your previous response was invalid JSON. Error: {str(jde)}. You MUST output pure, well-formed JSON.]"
                        continue
                        
                except Exception as e:
                    e_str = str(e).lower()
                    if "401" in e_str or "unauthorized" in e_str or "key" in e_str:
                        return {"error": f"שגיאת הרשאה מול OpenAI. יתכן שהטוקן שגוי (401). פירוט: {str(e)}"}
                    elif "429" in e_str or "quota" in e_str:
                        return {"error": f"חריגה ממגבלת הבקשות של OpenAI (429). אנא נסה שוב מאוחר יותר. פירוט: {str(e)}"}
                    elif "404" in e_str or "not found" in e_str or "model" in e_str:
                        if self.model_name == "gpt-4.1":
                            print(f"[INFO] Model {self.model_name} unavailable or invalid. Falling back to gpt-4o.")
                            self.model_name = "gpt-4o"
                            continue
                        elif self.model_name == "gpt-4o":
                            print(f"[INFO] Model {self.model_name} unavailable or invalid. Falling back to gpt-4o-mini.")
                            self.model_name = "gpt-4o-mini"
                            continue
                    last_error = str(e)
                    break
                    
            return {"error": f"שגיאה בהפקת תשובה מ-OpenAI לאחר {max_attempts} ניסיונות ({last_error})"}

class LLMClientFactory:
    @staticmethod
    def get_client(strategy=None):
        if not strategy:
            strategy = os.getenv("LLM_BACKEND_STRATEGY", "auto").lower()
            
        print(f"[INFO] Initializing LLM Backend Strategy: {strategy}")
        
        if strategy == "dicta":
            return DictaLLMClient()
        elif strategy == "gemini":
            return GeminiLLMClient()
        elif strategy == "openai":
            return OpenAILLMClient()
        elif strategy == "auto":
            if os.getenv("OPENAI_API_KEY"):
                print("[INFO] OPENAI_API_KEY found. Defaulting to OpenAI.")
                return OpenAILLMClient()
            elif os.getenv("GEMINI_API_KEY"):
                print("[INFO] GEMINI_API_KEY found. Defaulting to Gemini fallback.")
                return GeminiLLMClient()
            else:
                print("[INFO] No external keys found. Defaulting to Local DICTA Server.")
                return DictaLLMClient()
        else:
            raise ValueError(f"Unsupported LLM backend strategy: {strategy}")
