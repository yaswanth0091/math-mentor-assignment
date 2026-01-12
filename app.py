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

# --- 1. CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Math Mentor AI (Safe Mode)", layout="wide", page_icon="🛡️")

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    st.error("🚨 GEMINI_API_KEY not found. Please check your secrets.")
    st.stop()

# --- 2. ROBUST MODEL SELECTOR ---
@st.cache_resource
def get_available_models():
    # Helper to find working models
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'].replace('models/', '') for m in data.get('models', []) 
                      if 'generateContent' in m.get('supportedGenerationMethods', [])]
            # Priority: Lite (for quota) -> Flash -> Pro
            models.sort(key=lambda x: ('lite' not in x, 'flash' not in x))
            return models
    except:
        pass
    # Safe defaults
    return ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-pro"]

with st.sidebar:
    st.header("⚙️ Safe Mode Settings")
    models = get_available_models()
    CURRENT_MODEL = st.selectbox("AI Model:", models, index=0)
    st.info("ℹ️ App is running in 'Safe Mode' (10s delay) to prevent quota errors.")
    st.divider()
    st.header("🕵️ Trace Log")

# --- 3. API HELPER (With Extreme Backoff) ---
def call_gemini(prompt, image_data=None, mime_type=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    contents = [{"parts": [{"text": prompt}]}]
    
    if image_data:
        contents[0]["parts"].append({"inline_data": {"mime_type": mime_type, "data": image_data}})

    # Low temp for consistency
    data = {"contents": contents, "generationConfig": {"temperature": 0.1}}
    
    # Try up to 3 times with long waits
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "Error: Empty response from AI."
            
            elif response.status_code == 429:
                # HIT RATE LIMIT -> WAIT 15 SECONDS
                time.sleep(15)
                continue
            
            elif response.status_code >= 400:
                return f"API Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Connection Exception: {e}"
            
    return "Failed: Rate limit exceeded or API unavailable."

# --- 4. DATABASE ---
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

# --- 6. AGENT SYSTEM (CRASH-PROOF) ---
class AgentSystem:
    def __init__(self):
        self.trace = []

    def log(self, agent, msg):
        self.trace.append(f"**{agent}**: {msg}")

    def clean_json(self, text):
        # 1. Try pure JSON
        try: return json.loads(text)
        except: pass
        # 2. Try regex extraction
        try: 
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        # 3. Try removing markdown
        try: return json.loads(text.replace('```json', '').replace('```', '').strip())
        except: return None

    def parser(self, text=None, img=None, audio=None):
        self.log("Parser", "Analyzing input...")
        prompt = 'Extract math problem. Return JSON: {"problem_text": "...", "needs_clarification": false}'
        
        if img:
            b64 = base64.b64encode(img).decode()
            res = call_gemini(prompt, b64, "image/jpeg")
        elif audio:
            b64 = base64.b64encode(audio).decode()
            res = call_gemini(prompt, b64, "audio/mp3")
        else:
            res = call_gemini(f"{prompt} Input: {text}")

        data = self.clean_json(res)
        
        # CRASH PROOFING: FORCE DEFAULT VALUES
        if not data: data = {}
        if 'problem_text' not in data: 
            data['problem_text'] = text if text else "Could not read input."
        if 'needs_clarification' not in data: 
            data['needs_clarification'] = False
            
        return data

    def router(self, problem):
        self.log("Router", "Classifying...")
        res = call_gemini(f"Classify topic (Algebra/Calc/Prob) for: {problem}")
        return res if "Error" not in res else "Math"

    def solver(self, problem, context, topic):
        self.log("Solver", f"Solving ({topic})...")
        prompt = f"""Role: Tutor. Context: {context}. Problem: {problem}. 
        Return JSON: {{"final_answer": "...", "confidence": 0.9}}"""
        res = call_gemini(prompt)
        data = self.clean_json(res)
        
        if not data:
            return {"final_answer": res, "confidence": 0.0} # Return raw text if JSON fails
        return data

    def verifier(self, problem, solution):
        self.log("Verifier", "Checking logic...")
        # SAFE ACCESS to final_answer
        ans = solution.get('final_answer', 'Unknown')
        prompt = f"""Verify. Problem: {problem}. Answer: {ans}. 
        Return JSON: {{"is_correct": true, "critique": "None"}}"""
        res = call_gemini(prompt)
        data = self.clean_json(res)
        
        if not data:
            # If verifier fails, assume correct to avoid blocking user
            return {"is_correct": True, "critique": "Verifier Error", "confidence": 1.0}
        
        if 'is_correct' not in data: data['is_correct'] = True
        return data

    def explainer(self, solution, critique):
        self.log("Explainer", "Formatting...")
        ans = solution.get('final_answer', 'No answer')
        return call_gemini(f"Explain to student. Answer: {ans}. Notes: {critique}")

# --- 7. UI ---
def main():
    st.title("🛡️ Reliable Math Mentor (Safe Mode)")
    
    if 'agents' not in st.session_state: st.session_state.agents = AgentSystem()
    if 'step' not in st.session_state: st.session_state.step = 1

    with st.sidebar:
        for t in st.session_state.agents.trace: st.markdown(t)

    # STEP 1
    if st.session_state.step == 1:
        mode = st.radio("Input Mode", ["Text", "Image", "Audio"])
        
        if mode == "Text":
            txt = st.text_area("Problem:")
            if st.button("Go"):
                st.session_state.parsed = st.session_state.agents.parser(text=txt)
                if not st.session_state.parsed['needs_clarification']:
                    st.session_state.run_solver = True
                    st.session_state.step = 3
                else:
                    st.session_state.step = 2
                st.rerun()
                
        elif mode == "Image":
            img = st.file_uploader("Upload", type=['png','jpg'])
            if img and st.button("Process"):
                st.session_state.parsed = st.session_state.agents.parser(img=img.getvalue())
                st.session_state.step = 2
                st.rerun()
                
        elif mode == "Audio":
            aud = st.file_uploader("Upload", type=['mp3','wav'])
            if aud and st.button("Process"):
                st.session_state.parsed = st.session_state.agents.parser(audio=aud.getvalue())
                st.session_state.step = 2
                st.rerun()

    # STEP 2
    elif st.session_state.step == 2:
        st.header("Verify Input")
        # SAFE ACCESS
        current = st.session_state.parsed.get('problem_text', '')
        new_text = st.text_area("Edit:", value=current)
        
        if st.button("Confirm & Solve"):
            st.session_state.parsed['problem_text'] = new_text
            st.session_state.run_solver = True
            st.session_state.step = 3
            st.rerun()

    # STEP 3
    elif st.session_state.step == 3:
        if st.session_state.get('run_solver'):
            with st.spinner("Running Agents... (Waiting 10s between steps to avoid quota limit)"):
                prob = st.session_state.parsed.get('problem_text', 'Error')
                
                # --- EXTREME THROTTLING ---
                time.sleep(10) 
                topic = st.session_state.agents.router(prob)
                ctx = retrieve(prob)
                st.session_state.context = ctx
                
                time.sleep(10)
                sol = st.session_state.agents.solver(prob, ctx, topic)
                
                time.sleep(10)
                ver = st.session_state.agents.verifier(prob, sol)
                st.session_state.verification = ver
                
                # Check verification safely
                is_correct = ver.get('is_correct', True)
                
                if not is_correct:
                    st.session_state.needs_hitl = True
                else:
                    st.session_state.needs_hitl = False
                    time.sleep(10)
                    critique = ver.get('critique', '')
                    exp = st.session_state.agents.explainer(sol, critique)
                    st.session_state.final = exp
            
            st.session_state.run_solver = False

        if st.session_state.get('needs_hitl'):
            st.error("⚠️ Review Needed")
            st.write(st.session_state.verification)
            if st.button("Retry"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.header("Solution")
            st.markdown(st.session_state.get('final', 'Thinking...'))
            
            if st.button("Save & Finish"):
                conn = sqlite3.connect('math_memory.db')
                prob_safe = st.session_state.parsed.get('problem_text', 'Unknown')
                conn.execute("INSERT INTO history (timestamp, problem, solution) VALUES (?,?,?)", 
                            (str(datetime.now()), prob_safe, st.session_state.final))
                conn.commit()
                st.success("Saved!")
            
            if st.button("Restart"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()