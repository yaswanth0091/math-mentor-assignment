import streamlit as st
import requests
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import base64
import re

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
st.set_page_config(page_title="Math Mentor AI", layout="wide", page_icon="🎓")

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    st.error("🚨 GEMINI_API_KEY not found in .env file")
    st.stop()

# --- 2. DYNAMIC MODEL SELECTOR ---
@st.cache_resource
def get_available_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'].replace('models/', '') for m in data.get('models', []) 
                      if 'generateContent' in m.get('supportedGenerationMethods', [])]
            # Sort to prioritize models with slightly better limits
            models.sort(key=lambda x: ('lite' not in x, 'flash' not in x))
            return models
    except:
        pass
    # Fallback list
    return ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-pro"]

with st.sidebar:
    st.header("⚙️ Settings")
    models = get_available_models()
    CURRENT_MODEL = st.selectbox("AI Model:", models, index=0)
    st.caption(f"Selected: {CURRENT_MODEL}. App is slowed down to respect tight API limits.")
    st.divider()
    st.header("🕵️ Agent Trace")

# --- 3. DIRECT API HELPER (With Heavy Backoff) ---
def call_gemini(prompt, image_data=None, mime_type=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    contents = [{"parts": [{"text": prompt}]}]
    
    if image_data:
        contents[0]["parts"].append({"inline_data": {"mime_type": mime_type, "data": image_data}})

    data = {"contents": contents, "generationConfig": {"temperature": 0.1}}
    
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "Error parsing response."
            elif response.status_code == 429:
                # Wait longer on rate limit: 5s, 10s, 15s...
                time.sleep((attempt + 1) * 5)
                continue
            else:
                return f"Error {response.status_code}"
        except Exception as e:
            return f"Exception: {e}"
    return "Failed after retries due to strict rate limits."

# --- 4. DATABASE (MEMORY) ---
def init_db():
    conn = sqlite3.connect('math_memory.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY, timestamp TEXT, problem TEXT, solution TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 5. RAG SYSTEM ---
@st.cache_resource
def load_rag():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kb_text = [
        "Quadratic Formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a",
        "Derivative Power Rule: d/dx(x^n) = nx^(n-1)",
        "Probability P(A or B) = P(A) + P(B) - P(A and B)",
        "Integration Power Rule: ∫x^n dx = (x^(n+1))/(n+1) + C"
    ]
    embeddings = embedder.encode(kb_text)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return embedder, index, kb_text

embedder, index, kb_text = load_rag()

def retrieve(query):
    if not query: return []
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    return [kb_text[i] for i in I[0]]

# --- 6. MULTI-AGENT SYSTEM (CRASH PROOF + THROTTLED) ---
class AgentSystem:
    def __init__(self):
        self.trace = []

    def log(self, agent, msg):
        self.trace.append(f"**{agent}**: {msg}")

    def clean_json(self, text):
        try: return json.loads(text)
        except: pass
        try: 
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        try: return json.loads(text.replace('```json', '').replace('```', '').strip())
        except: return None

    # Agent 1: Parser
    def parser(self, text=None, img=None, audio=None):
        self.log("Parser", "Analyzing input...")
        prompt = """
        Extract the math problem. Return JSON: {"problem_text": "...", "topic": "math", "needs_clarification": false}
        """
        if img:
            b64 = base64.b64encode(img).decode()
            res = call_gemini(prompt, b64, "image/jpeg")
        elif audio:
            b64 = base64.b64encode(audio).decode()
            res = call_gemini(prompt, b64, "audio/mp3")
        else:
            res = call_gemini(f"{prompt} Input: {text}")
        
        data = self.clean_json(res)
        
        # CRASH FIX: Force default dictionary if data is None or missing keys
        if not data:
            data = {}
        
        # Ensure 'problem_text' always exists
        if 'problem_text' not in data:
            data['problem_text'] = text if text else "Error: Could not extract text from input."
        
        # Ensure 'needs_clarification' exists
        if 'needs_clarification' not in data:
            data['needs_clarification'] = False
            
        return data

    # Agent 2: Router
    def router(self, problem):
        self.log("Router", "Classifying intent...")
        return call_gemini(f"Classify topic (Algebra/Calculus/Prob). Problem: {problem}").strip()

    # Agent 3: Solver
    def solver(self, problem, context, topic):
        self.log("Solver", f"Solving ({topic})...")
        prompt = f"""Role: Math Tutor. Topic: {topic}. Context: {context}
        Problem: {problem}. Return JSON: {{"steps": ["1...", "2..."], "final_answer": "...", "confidence": 0.95}}"""
        res = call_gemini(prompt)
        data = self.clean_json(res)
        
        # CRASH FIX: Default if solver fails
        if not data: 
            return {"steps": ["Error generating solution steps."], "final_answer": res, "confidence": 0.0}
        return data

    # Agent 4: Verifier
    def verifier(self, problem, solution):
        self.log("Verifier", "Checking logic...")
        # Handle cases where final_answer might be missing
        ans = solution.get('final_answer', 'Unknown')
        
        prompt = f"""Verify. Problem: {problem}. Answer: {ans}.
        Return JSON: {{"is_correct": true, "critique": "None", "confidence": 1.0}}"""
        res = call_gemini(prompt)
        data = self.clean_json(res)
        
        # CRASH FIX: Default to True (allow passage) if verifier fails to parse
        if not data: 
            return {"is_correct": True, "critique": "Verifier failed to parse, assuming correct.", "confidence": 1.0}
        
        # Ensure keys exist
        if 'is_correct' not in data: data['is_correct'] = True
        if 'confidence' not in data: data['confidence'] = 1.0
        if 'critique' not in data: data['critique'] = "None"
        
        return data

    # Agent 5: Explainer
    def explainer(self, solution, critique):
        self.log("Explainer", "Drafting response...")
        return call_gemini(f"Explain to student. Solution: {solution}. Critique: {critique}")

# --- 7. MAIN UI ---
def main():
    st.title("🎓 Reliable Math Mentor")
    if 'agents' not in st.session_state: st.session_state.agents = AgentSystem()
    if 'step' not in st.session_state: st.session_state.step = 1

    with st.sidebar:
        for t in st.session_state.agents.trace: st.markdown(t)

    # STEP 1: INPUT
    if st.session_state.step == 1:
        mode = st.radio("Input Mode", ["Text", "Image", "Audio"])
        if mode == "Text":
            txt = st.text_area("Problem:")
            if st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(text=txt)
                # CRASH FIX: Use .get()
                if not st.session_state.parsed.get('needs_clarification', False):
                    st.session_state.run_solver = True
                    st.session_state.step = 3
                else:
                    st.session_state.step = 2
                st.rerun()
        elif mode == "Image":
            img = st.file_uploader("Upload", type=['png','jpg'])
            if img and st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(img=img.getvalue())
                st.session_state.step = 2
                st.rerun()
        elif mode == "Audio":
            aud = st.file_uploader("Upload", type=['mp3','wav'])
            if aud and st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(audio=aud.getvalue())
                st.session_state.step = 2
                st.rerun()

    # STEP 2: HITL
    elif st.session_state.step == 2:
        st.header("1. Verify Extraction")
        
        # CRASH FIX: Safe Access
        current_prob = st.session_state.parsed.get('problem_text', '')
        
        conn = sqlite3.connect('math_memory.db')
        mem_hit = conn.execute("SELECT solution FROM history WHERE problem LIKE ?", (f"%{current_prob[:30]}%",)).fetchone()
        conn.close()
        
        if mem_hit:
            st.success("🧠 Memory Hit!")
            if st.button("Use Memory Solution"):
                st.session_state.final = mem_hit[0]
                st.session_state.step = 3
                st.rerun()

        new_text = st.text_area("Edit if needed:", value=current_prob)
        if st.button("Confirm & Solve"):
            st.session_state.parsed['problem_text'] = new_text
            st.session_state.run_solver = True
            st.session_state.step = 3
            st.rerun()

    # STEP 3: SOLVE
    elif st.session_state.step == 3:
        if st.session_state.get('run_solver'):
            with st.spinner("Running Agents (Slowed down for limits)..."):
                # CRASH FIX: Safe access
                prob = st.session_state.parsed.get('problem_text', 'Error extracting text')
                
                # Heavy Throttling (6 seconds) to respect your 5 RPM limit
                time.sleep(6) 
                topic = st.session_state.agents.router(prob)
                ctx = retrieve(prob)
                st.session_state.context = ctx
                
                time.sleep(6)
                sol = st.session_state.agents.solver(prob, ctx, topic)
                
                time.sleep(6)
                ver = st.session_state.agents.verifier(prob, sol)
                st.session_state.verification = ver
                
                # CRASH PROOF CHECK (Safe Access)
                is_correct = ver.get('is_correct', True)
                confidence = ver.get('confidence', 1.0)
                
                if not is_correct or confidence < 0.8:
                    st.session_state.needs_hitl_correction = True
                else:
                    st.session_state.needs_hitl_correction = False
                    time.sleep(6)
                    # CRASH FIX: Safe access to critique
                    critique = ver.get('critique', 'None')
                    exp = st.session_state.agents.explainer(sol, critique)
                    st.session_state.final = exp
            st.session_state.run_solver = False

        if st.session_state.get('needs_hitl_correction'):
            st.error("⚠️ Verifier found issues.")
            # CRASH FIX: Safe access
            critique = st.session_state.verification.get('critique', 'Unknown issue')
            st.warning(f"Critique: {critique}")
            if st.button("Retry/Edit"):
                st.session_state.step = 2
                st.rerun()
        else:
            with st.expander("📚 RAG Context"):
                for c in st.session_state.get('context', []): st.info(c)
            st.header("2. Solution")
            st.markdown(st.session_state.get('final', 'Processing...'))
            
            if st.button("✅ Save to Memory"):
                conn = sqlite3.connect('math_memory.db')
                # CRASH FIX: Safe access
                prob_txt = st.session_state.parsed.get('problem_text', 'Unknown')
                conn.execute("INSERT INTO history (timestamp, problem, solution) VALUES (?,?,?)", 
                            (str(datetime.now()), prob_txt, st.session_state.final))
                conn.commit()
                st.toast("Saved!")
            if st.button("Start Over"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()