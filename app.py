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

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
st.set_page_config(page_title="Math Mentor AI", layout="wide", page_icon="🎓")

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    st.error("🚨 GEMINI_API_KEY not found in .env file")
    st.stop()

# --- 2. DYNAMIC MODEL SELECTOR (Updated based on your screenshot) ---
@st.cache_resource
def get_available_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'].replace('models/', '') for m in data.get('models', []) 
                      if 'generateContent' in m.get('supportedGenerationMethods', [])]
            # Sort to prioritize models with slightly better limits if available
            models.sort(key=lambda x: ('lite' not in x, 'flash' not in x))
            return models
    except:
        pass
    # Fallback list based on your screenshot
    return [
        "gemini-2.5-flash-lite", # Try for 10 RPM limit
        "gemini-2.5-flash",      # 5 RPM limit
        "gemini-1.5-flash",      # Standard free tier (usually better limits)
        "gemini-pro"
    ]

with st.sidebar:
    st.header("⚙️ Settings")
    models = get_available_models()
    # Default to index 0 (hopefully the 'lite' model for better RPM)
    CURRENT_MODEL = st.selectbox("AI Model:", models, index=0)
    st.caption(f"Selected: {CURRENT_MODEL}. App is slowed down to respect tight API limits.")
    st.divider()
    st.header("🕵️ Agent Trace")

# --- 3. DIRECT API HELPER (With heavy retry backoff) ---
def call_gemini(prompt, image_data=None, mime_type=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    contents = [{"parts": [{"text": prompt}]}]
    
    if image_data:
        contents[0]["parts"].append({"inline_data": {"mime_type": mime_type, "data": image_data}})

    data = {"contents": contents, "generationConfig": {"temperature": 0.1}}
    
    # Increased retries with exponential backoff for strict limits
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "Error parsing response."
            elif response.status_code == 429:
                # Wait significantly longer on rate limit hit: 5s, 10s, 15s...
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time) 
                continue
            else:
                return f"Error {response.status_code} - {response.text}"
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
        "Discriminant D = b^2 - 4ac. D>0 (2 real), D=0 (1 real), D<0 (complex)",
        "Derivative Power Rule: d/dx(x^n) = nx^(n-1)",
        "Derivative of sin(x) is cos(x), cos(x) is -sin(x)",
        "Integration Power Rule: ∫x^n dx = (x^(n+1))/(n+1) + C",
        "Integration by Parts: ∫udv = uv - ∫vdu",
        "Probability P(A or B) = P(A) + P(B) - P(A and B)",
        "Independent Events P(A and B) = P(A)*P(B)",
        "Logarithm Product: log(ab) = log(a) + log(b)",
        "Euler's Identity: e^(ix) = cos(x) + i*sin(x)"
    ]
    embeddings = embedder.encode(kb_text)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return embedder, index, kb_text

embedder, index, kb_text = load_rag()

def retrieve(query):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    return [kb_text[i] for i in I[0]]

# --- 6. MULTI-AGENT SYSTEM ---
import re
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
        Extract the math problem. If text is clear math, set "needs_clarification": false.
        Return JSON: {"problem_text": "...", "topic": "math", "needs_clarification": false}
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
        if not data: return {"problem_text": text or "Error", "topic": "Unknown", "needs_clarification": True}
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
        if not data: return {"steps": ["Error"], "final_answer": res, "confidence": 0.0}
        return data

    # Agent 4: Verifier
    def verifier(self, problem, solution):
        self.log("Verifier", "Checking logic...")
        prompt = f"""Verify. Problem: {problem}. Answer: {solution.get('final_answer')}.
        Return JSON: {{"is_correct": true, "critique": "None", "confidence": 1.0}}"""
        res = call_gemini(prompt)
        data = self.clean_json(res)
        if not data: return {"is_correct": False, "critique": "Parsing Error", "confidence": 0.0}
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
                # Smart Skip HITL for high confidence text
                if not st.session_state.parsed.get('needs_clarification'):
                    st.session_state.run_solver = True
                    st.session_state.step = 3
                else:
                    st.session_state.step = 2
                st.rerun()
        elif mode == "Image":
            img = st.file_uploader("Upload", type=['png','jpg'])
            if img and st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(img=img.getvalue())
                st.session_state.step = 2 # Mandatory HITL for media
                st.rerun()
        elif mode == "Audio":
            aud = st.file_uploader("Upload", type=['mp3','wav'])
            if aud and st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(audio=aud.getvalue())
                st.session_state.step = 2 # Mandatory HITL for media
                st.rerun()

    # STEP 2: HITL & MEMORY
    elif st.session_state.step == 2:
        st.header("1. Verify Extraction")
        
        # Memory check
        conn = sqlite3.connect('math_memory.db')
        mem_hit = conn.execute("SELECT solution FROM history WHERE problem LIKE ?", (f"%{st.session_state.parsed.get('problem_text','')[:30]}%",)).fetchone()
        conn.close()
        if mem_hit:
            st.success("🧠 Memory Hit!")
            if st.button("Use Memory Solution"):
                st.session_state.final = mem_hit[0]
                st.session_state.agents.log("Memory", "Used cached solution")
                st.session_state.step = 3
                st.rerun()

        new_text = st.text_area("Edit if needed:", value=st.session_state.parsed.get('problem_text', ''))
        if st.button("Confirm & Solve"):
            st.session_state.parsed['problem_text'] = new_text
            st.session_state.run_solver = True
            st.session_state.step = 3
            st.rerun()

    # STEP 3: SOLVE & OUTPUT
    elif st.session_state.step == 3:
        if st.session_state.get('run_solver'):
            # HEAVY THROTTLING FOR 5 RPM LIMIT
            with st.spinner("Running Agents (Slowed down for strict API limits)..."):
                prob = st.session_state.parsed['problem_text']
                # Wait 6 seconds between calls to stay under 10 RPM, safer for 5 RPM
                time.sleep(6) 
                topic = st.session_state.agents.router(prob)
                ctx = retrieve(prob)
                st.session_state.context = ctx
                time.sleep(6)
                sol = st.session_state.agents.solver(prob, ctx, topic)
                time.sleep(6)
                ver = st.session_state.agents.verifier(prob, sol)
                st.session_state.verification = ver
                
                if not ver['is_correct'] or ver.get('confidence', 1.0) < 0.8:
                    st.session_state.needs_hitl_correction = True
                else:
                    st.session_state.needs_hitl_correction = False
                    time.sleep(6)
                    exp = st.session_state.agents.explainer(sol, ver['critique'])
                    st.session_state.final = exp
            st.session_state.run_solver = False

        if st.session_state.get('needs_hitl_correction'):
            st.error("⚠️ Verifier found issues.")
            st.warning(f"Critique: {st.session_state.verification['critique']}")
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
                conn.execute("INSERT INTO history (timestamp, problem, solution) VALUES (?,?,?)", 
                            (str(datetime.now()), st.session_state.parsed['problem_text'], st.session_state.final))
                conn.commit()
                st.toast("Saved!")
            if st.button("Start Over"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()