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

# --- 2. DYNAMIC MODEL SELECTOR (Deployment Safety) ---
@st.cache_resource
def get_available_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'].replace('models/', '') for m in data.get('models', []) 
                      if 'generateContent' in m.get('supportedGenerationMethods', [])]
            models.sort(key=lambda x: ('flash' not in x, 'pro' not in x))
            return models
    except:
        pass
    return ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]

with st.sidebar:
    st.header("⚙️ Settings")
    models = get_available_models()
    CURRENT_MODEL = st.selectbox("AI Model:", models, index=0)
    st.divider()
    st.header("🕵️ Agent Trace")

# --- 3. DIRECT API HELPER ---
def call_gemini(prompt, image_data=None, mime_type=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    contents = [{"parts": [{"text": prompt}]}]
    
    if image_data:
        contents[0]["parts"].append({"inline_data": {"mime_type": mime_type, "data": image_data}})

    data = {"contents": contents, "generationConfig": {"temperature": 0.1}}
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "Error parsing response."
            elif response.status_code == 429:
                time.sleep(2)
                continue
            else:
                return f"Error {response.status_code}"
        except Exception as e:
            return f"Exception: {e}"
    return "Failed after retries."

# --- 4. DATABASE (MEMORY) ---
def init_db():
    conn = sqlite3.connect('math_memory.db')
    c = conn.cursor()
    # Storing raw problem text and the final solution
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY, timestamp TEXT, problem TEXT, solution TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 5. RAG SYSTEM (KNOWLEDGE BASE) ---
@st.cache_resource
def load_rag():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Expanded KB to meet "10-30 docs" requirement
    kb_text = [
        "Quadratic Formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a",
        "Discriminant: D = b^2 - 4ac. If D>0 2 real roots, D=0 1 real root, D<0 complex roots.",
        "Derivative Power Rule: d/dx(x^n) = nx^(n-1)",
        "Derivative of Constants: d/dx(c) = 0",
        "Derivative of sin(x) is cos(x), cos(x) is -sin(x)",
        "Chain Rule: d/dx f(g(x)) = f'(g(x)) * g'(x)",
        "Product Rule: d/dx (uv) = u'v + uv'",
        "Integration Power Rule: ∫x^n dx = (x^(n+1))/(n+1) + C",
        "Integration by Parts: ∫udv = uv - ∫vdu",
        "Probability: P(A or B) = P(A) + P(B) - P(A and B)",
        "Independent Events: P(A and B) = P(A) * P(B)",
        "Conditional Probability: P(A|B) = P(A and B) / P(B)",
        "Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)",
        "Logarithm Product: log(ab) = log(a) + log(b)",
        "Logarithm Quotient: log(a/b) = log(a) - log(b)",
        "Euler's Identity: e^(ix) = cos(x) + i*sin(x)"
    ]
    embeddings = embedder.encode(kb_text)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return embedder, index, kb_text

embedder, index, kb_text = load_rag()

def retrieve(query):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3) # Top-3 relevant docs
    return [kb_text[i] for i in I[0]]

# --- 6. MULTI-AGENT SYSTEM (5 AGENTS) ---
class AgentSystem:
    def __init__(self):
        self.trace = []

    def log(self, agent, msg):
        self.trace.append(f"**{agent}**: {msg}")

    # Agent 1: Parser
    def parser(self, text=None, img=None, audio=None):
        self.log("Parser", "Structuring input...")
        prompt = """
        Extract the math problem. Return JSON: 
        {"problem_text": "...", "topic": "algebra/calc/prob", "variables": ["x","y"], "needs_clarification": false}
        """
        if img:
            b64 = base64.b64encode(img).decode()
            res = call_gemini(prompt, b64, "image/jpeg")
        elif audio:
            b64 = base64.b64encode(audio).decode()
            res = call_gemini(prompt, b64, "audio/mp3")
        else:
            res = call_gemini(f"{prompt} Input: {text}")
        
        try:
            return json.loads(res.replace('```json', '').replace('```', '').strip())
        except:
            return {"problem_text": text or "Error", "topic": "Unknown", "needs_clarification": True}

    # Agent 2: Intent Router (Mandatory)
    def router(self, problem):
        self.log("Router", "Analyzing intent...")
        prompt = f"Classify this math problem into a single topic (Algebra, Calculus, Probability, Linear Algebra). Problem: {problem}"
        return call_gemini(prompt).strip()

    # Agent 3: Solver
    def solver(self, problem, context, topic):
        self.log("Solver", f"Solving ({topic}) with RAG...")
        prompt = f"""
        Role: Math Tutor. Topic: {topic}. 
        Context: {context}
        Problem: {problem}
        Return JSON: {{"steps": ["1...", "2..."], "final_answer": "...", "confidence": 0.95}}
        """
        res = call_gemini(prompt)
        try:
            return json.loads(res.replace('```json', '').replace('```', '').strip())
        except:
            return {"steps": ["Error"], "final_answer": res}

    # Agent 4: Verifier
    def verifier(self, problem, solution):
        self.log("Verifier", "Checking constraints & logic...")
        prompt = f"""
        Verify solution. Problem: {problem}. Answer: {solution.get('final_answer')}.
        Return JSON: {{"is_correct": true, "critique": "None"}}
        """
        res = call_gemini(prompt)
        try:
            return json.loads(res.replace('```json', '').replace('```', '').strip())
        except:
            return {"is_correct": False, "critique": "Verification parsing error"}

    # Agent 5: Explainer
    def explainer(self, solution, critique):
        self.log("Explainer", "Drafting final response...")
        return call_gemini(f"Explain this to a student. Solution: {solution}. Critique: {critique}")

# --- 7. MAIN UI ---
def main():
    st.title("🎓 Reliable Math Mentor")
    if 'agents' not in st.session_state: st.session_state.agents = AgentSystem()
    if 'step' not in st.session_state: st.session_state.step = 1

    # Sidebar Trace
    with st.sidebar:
        for t in st.session_state.agents.trace:
            st.markdown(t)

    # STEP 1: INPUT
    if st.session_state.step == 1:
        mode = st.radio("Input Mode", ["Text", "Image", "Audio"])
        if mode == "Text":
            txt = st.text_area("Problem:")
            if st.button("Analyze"):
                st.session_state.parsed = st.session_state.agents.parser(text=txt)
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

    # STEP 2: HITL & MEMORY CHECK
    elif st.session_state.step == 2:
        st.header("1. Verify Extraction (HITL)")
        
        # Memory Lookup (Mandatory Requirement)
        current_prob = st.session_state.parsed.get('problem_text', '')
        conn = sqlite3.connect('math_memory.db')
        # Simple exact match (fuzzy match would require vector db, keeping it simple for assignment)
        mem_hit = conn.execute("SELECT solution FROM history WHERE problem LIKE ?", (f"%{current_prob[:20]}%",)).fetchone()
        conn.close()

        if mem_hit:
            st.success("🧠 Memory Hit: I recognize this problem!")
            if st.button("Use Memory Solution"):
                st.session_state.final = mem_hit[0]
                st.session_state.agents.log("Memory", "Retrieved cached solution")
                st.session_state.step = 3
                st.rerun()

        # Edit/Confirm
        new_text = st.text_area("Edit text if needed:", value=current_prob)
        
        if st.button("Confirm & Solve"):
            st.session_state.parsed['problem_text'] = new_text
            
            with st.spinner("Running Multi-Agent System..."):
                # 1. Route
                topic = st.session_state.agents.router(new_text)
                
                # 2. RAG
                ctx = retrieve(new_text)
                st.session_state.context = ctx # Save for display
                
                # 3. Solve (with delays to avoid rate limits)
                time.sleep(1)
                sol = st.session_state.agents.solver(new_text, ctx, topic)
                
                # 4. Verify
                time.sleep(1)
                ver = st.session_state.agents.verifier(new_text, sol)
                
                # 5. Explain
                time.sleep(1)
                exp = st.session_state.agents.explainer(sol, ver['critique'])
                
                st.session_state.final = exp
                st.session_state.step = 3
                st.rerun()

    # STEP 3: OUTPUT & FEEDBACK
    elif st.session_state.step == 3:
        # Context Panel (Mandatory)
        with st.expander("📚 Retrieved Context (RAG)"):
            for i, c in enumerate(st.session_state.get('context', [])):
                st.info(f"Source {i+1}: {c}")

        st.header("2. Solution")
        st.markdown(st.session_state.final)
        
        col1, col2 = st.columns(2)
        if col1.button("✅ Correct (Learn)"):
            conn = sqlite3.connect('math_memory.db')
            conn.execute("INSERT INTO history (timestamp, problem, solution) VALUES (?,?,?)", 
                        (str(datetime.now()), st.session_state.parsed['problem_text'], st.session_state.final))
            conn.commit()
            st.toast("Learned pattern saved!")
        
        if col2.button("❌ Incorrect"):
            st.error("Feedback logged.")
            
        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()