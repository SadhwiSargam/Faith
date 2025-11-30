
# """
# Streamlit Mood Journal — Multi-Agent Emotional Companion (Gemini)

# Before running:
#  - set GEMINI_API_KEY environment variable
#    Windows: setx GEMINI_API_KEY "<your_key>"
#    Linux/Mac: export GEMINI_API_KEY="<your_key>"
#  - pip install -r requirements.txt
#  - streamlit run app.py
# """

# import os
# import json
# import time
# from datetime import datetime
# from typing import List, Dict, Any

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv

# load_dotenv()

# # ---------------------------
# # Gemini SDK setup
# # ---------------------------
# from google import genai

# GENAI_KEY = os.getenv("GEMINI_API_KEY")
# if not GENAI_KEY:
#     st.error("GEMINI_API_KEY environment variable is missing. Set it before running the app.")
#     st.stop()

# # pick a model you have access to
# MODEL_NAME = "gemini-2.5-flash"
# client = genai.Client(api_key=GENAI_KEY)

# MEMORY_FILE = "memory_store.json"

# # ---------------------------
# # Persistent memory helpers
# # ---------------------------
# def load_memory() -> List[Dict[str, Any]]:
#     if not os.path.exists(MEMORY_FILE):
#         return []
#     try:
#         with open(MEMORY_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return []

# def save_memory(entries: List[Dict[str, Any]]):
#     # Ensure timestamps are strings before saving
#     serializable = []
#     for e in entries:
#         copy_e = dict(e)
#         ts = copy_e.get("timestamp")
#         if isinstance(ts, (datetime, pd.Timestamp)):
#             try:
#                 copy_e["timestamp"] = ts.isoformat()
#             except Exception:
#                 copy_e["timestamp"] = datetime.now().astimezone().isoformat()
#         serializable.append(copy_e)
#     with open(MEMORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(serializable, f, ensure_ascii=False, indent=2)

# # ---------------------------
# # Model call helper
# # ---------------------------
# def call_model(prompt: str) -> Dict[str, Any]:
#     """
#     Calls the Gemini model and returns {"success": bool, "text": str}.
#     Tries twice before failing gracefully.
#     """
#     for _ in range(2):
#         try:
#             response = client.models.generate_content(
#                 model=MODEL_NAME,
#                 # many SDK versions accept list-of-strings; if you get errors try contents=[{"type":"text","text":prompt}]
#                 contents=[prompt]
#             )
#             text = getattr(response, "text", None)
#             if text is None:
#                 # some SDK returns structured object — stringify safely
#                 text = str(response)
#             return {"success": True, "text": text}
#         except Exception as e:
#             time.sleep(1)
#     return {"success": False, "text": "Sorry, I couldn't process this entry. Please try again later."}

# # ---------------------------
# # AGENT 1: Intake (robust prompt)
# # ---------------------------
# INTAKE_PROMPT_TEMPLATE = """
# You are an analytical intake agent for a mood-journal app.
# The user entry is below. Produce ONLY a single JSON object (no explanation, no code fences)
# with these fields:

# - timestamp: current time in ISO format
# - text: the original entry (exact)
# - mood_label: one of (happy, sad, anxious, stressed, angry, calm, excited, overwhelmed, mixed, neutral)
# - valence: a number between -1.0 (very negative) and 1.0 (very positive)
# - arousal: a number between 0.0 (calm) and 1.0 (very activated)
# - emotions: list of up to 3 short emotion labels (strings)
# - triggers: list of up to 3 short trigger phrases
# - summary: 1 sentence compassionate summary
# - length: integer character count of the text

# User entry:
# \"\"\"{entry_text}\"\"\"
# """

# def extract_first_json(text: str) -> Dict[str, Any]:
#     # Find first {...} block and parse. Return {} if fails.
#     start = text.find("{")
#     end = text.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         try:
#             return json.loads(text[start:end+1])
#         except Exception:
#             pass
#     return {}

# def intake_agent(raw_text: str) -> Dict[str, Any]:
#     prompt = INTAKE_PROMPT_TEMPLATE.format(entry_text=raw_text)
#     res = call_model(prompt)
#     if not res["success"]:
#         return {
#             "timestamp": datetime.now().astimezone().isoformat(),
#             "text": raw_text,
#             "mood_label": "undetected",
#             "valence": 0.0,
#             "arousal": 0.5,
#             "emotions": [],
#             "triggers": [],
#             "summary": "Sorry, I couldn't analyze this entry.",
#             "length": len(raw_text),
#             "error": True,
#             "error_text": res["text"]
#         }
#     parsed = extract_first_json(res["text"])
#     if not parsed:
#         # fallback structure
#         parsed = {
#             "timestamp": datetime.now().astimezone().isoformat(),
#             "text": raw_text,
#             "mood_label": "undetected",
#             "valence": 0.0,
#             "arousal": 0.5,
#             "emotions": [],
#             "triggers": [],
#             "summary": raw_text.strip(),
#             "length": len(raw_text),
#             "error": True,
#             "error_text": res["text"][:1000]
#         }
#     # ensure fields exist and normalize timestamp to ISO string
#     ts = parsed.get("timestamp")
#     if not ts:
#         parsed["timestamp"] = datetime.now().astimezone().isoformat()
#     else:
#         # keep as string; if it's a datetime, convert
#         if isinstance(ts, (datetime, pd.Timestamp)):
#             parsed["timestamp"] = ts.isoformat()
#     parsed["text"] = parsed.get("text", raw_text)
#     parsed["length"] = int(parsed.get("length", len(parsed["text"])))
#     return parsed

# # ---------------------------
# # AGENT 2: Analysis
# # ---------------------------
# def analysis_agent(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
#     if not entries:
#         return {
#             "common_mood": None,
#             "avg_valence": 0.0,
#             "avg_arousal": 0.5,
#             "trigger_counts": {},
#             "weekly_summary": "No data."
#         }

#     moods, triggers, valences, arousals = {}, {}, [], []
#     for e in entries:
#         ml = e.get("mood_label", "unknown")
#         moods[ml] = moods.get(ml, 0) + 1
#         try:
#             valences.append(float(e.get("valence", 0.0)))
#         except Exception:
#             valences.append(0.0)
#         try:
#             arousals.append(float(e.get("arousal", 0.5)))
#         except Exception:
#             arousals.append(0.5)
#         for t in e.get("triggers", []):
#             triggers[t] = triggers.get(t, 0) + 1

#     avg_valence = sum(valences) / max(1, len(valences))
#     avg_arousal = sum(arousals) / max(1, len(arousals))
#     common_mood = max(moods.items(), key=lambda x: x[1])[0] if moods else None

#     prompt = f"""
# You are an Analysis Agent. Summarize the following journal entries (JSON list).
# Output ONLY a JSON object with these fields:
# - weekly_summary (2-4 sentences)
# - valence_trend (up/down/flat)
# - recommended_action (short practical suggestion)

# Data: {json.dumps(entries, ensure_ascii=False, indent=2)}
# """
#     raw = call_model(prompt)
#     raw_text = raw.get("text", "")
#     summary_parsed = extract_first_json(raw_text)
#     if not summary_parsed:
#         summary_parsed = {
#             "weekly_summary": "Could not generate summary.",
#             "valence_trend": "flat",
#             "recommended_action": ""
#         }

#     out = {
#         "common_mood": common_mood,
#         "avg_valence": avg_valence,
#         "avg_arousal": avg_arousal,
#         "trigger_counts": triggers,
#         "weekly_summary": summary_parsed.get("weekly_summary", "…"),
#         "valence_trend": summary_parsed.get("valence_trend", "flat"),
#         "recommended_action": summary_parsed.get("recommended_action", "")
#     }
#     return out

# # ---------------------------
# # AGENT 3: Support
# # ---------------------------
# def support_agent(latest_entry: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
#     prompt = f"""
# You are a supportive, poetic agent. Given the latest structured entry and the analysis,
# output ONLY a JSON object with:
# - empathetic_text (6-10 sentences, warm)
# - micro_actions (3 short practical actions)
# - reframe_prompt (one compassionate sentence)
# - breathing_exercise (one short guided breath)

# Latest entry: {json.dumps(latest_entry, ensure_ascii=False)}
# Analysis: {json.dumps(analysis, ensure_ascii=False)}
# """
#     raw = call_model(prompt)
#     raw_text = raw.get("text", "")
#     parsed = extract_first_json(raw_text)
#     if not parsed:
#         return {
#             "empathetic_text": "I hear you. Thank you for sharing.",
#             "micro_actions": [
#                 "Take 3 slow deep breaths and ground yourself.",
#                 "Write one sentence about what you need right now.",
#                 "Step outside for 5 minutes and notice three sounds."
#             ],
#             "reframe_prompt": "This feeling is a signal, not a verdict.",
#             "breathing_exercise": "Inhale 4s, hold 2s, exhale 6s; repeat 4 times."
#         }
#     return parsed

# # ---------------------------
# # Visualization helpers
# # ---------------------------
# def build_mood_df(entries: List[Dict[str, Any]]) -> pd.DataFrame:
#     rows = []
#     for e in entries:
#         ts_raw = e.get("timestamp")
#         try:
#             dt = pd.to_datetime(ts_raw, utc=True, errors="coerce")
#         except Exception:
#             dt = pd.Timestamp.utcnow().tz_localize("UTC")
#         if pd.isna(dt):
#             dt = pd.Timestamp.utcnow().tz_localize("UTC")
#         rows.append({
#             "timestamp": dt,
#             "mood_label": e.get("mood_label", "unknown"),
#             "valence": float(e.get("valence", 0.0)),
#             "arousal": float(e.get("arousal", 0.5))
#         })
#     if not rows:
#         return pd.DataFrame(columns=["timestamp", "mood_label", "valence", "arousal"])
#     df = pd.DataFrame(rows)
#     df = df.sort_values("timestamp")
#     return df

# def plot_trends(df: pd.DataFrame):
#     if df.empty:
#         st.info("No entries yet to visualize.")
#         return
#     fig, axes = plt.subplots(1, 2, figsize=(12, 3))
#     axes[0].plot(df["timestamp"].dt.tz_convert(None), df["valence"], marker="o")
#     axes[0].set_title("Valence (-1 → 1)")
#     axes[0].set_xlabel("Date")
#     axes[0].set_ylabel("Valence")
#     axes[0].grid(True)

#     axes[1].scatter(df["timestamp"].dt.tz_convert(None), df["arousal"])
#     axes[1].set_title("Arousal (calm → agitated)")
#     axes[1].set_xlabel("Date")
#     axes[1].set_ylabel("Arousal")
#     axes[1].grid(True)

#     st.pyplot(fig)

# # ---------------------------
# # Streamlit App Layout
# # ---------------------------
# st.set_page_config(page_title="Emotional Companion", layout="centered")
# st.title("🌿 Faith - Mood Journal")
# st.markdown("Write what you feel. Multi-agent AI provides structured insights, support, and trends.")

# # Default/sample text
# default_text = st.session_state.get("sample_text", "")
# user_text = st.text_area("Your Journal Entry:", value=default_text, height=180)

# col1, col2 = st.columns([1, 1])
# with col1:
#     if st.button("Submit Entry"):
#         if not user_text.strip():
#             st.warning("Please write something.")
#         else:
#             with st.spinner("Processing..."):
#                 structured = intake_agent(user_text)
#                 # ensure timestamp string
#                 if isinstance(structured.get("timestamp"), (datetime, pd.Timestamp)):
#                     structured["timestamp"] = structured["timestamp"].isoformat()
#                 memory = load_memory()
#                 memory.append(structured)
#                 save_memory(memory)
#                 st.success("Entry saved.")
#                 analysis = analysis_agent(memory)
#                 support = support_agent(structured, analysis)
#                 st.session_state.update(latest_structured=structured,
#                                         latest_analysis=analysis,
#                                         latest_support=support)
#                 st.session_state["sample_text"] = ""

#     if st.button("Clear All Entries"):
#         if os.path.exists(MEMORY_FILE):
#             os.remove(MEMORY_FILE)
#         st.session_state.clear()
#         st.success("All entries cleared.")

# with col2:
#     st.markdown("**Sample Entries:**")
#     if st.button("Happy but restless"):
#         st.session_state["sample_text"] = "I am happy but restless today. I have a lot on my mind but some small joys too."
#     if st.button("Overwhelmed at college"):
#         st.session_state["sample_text"] = "I'm overwhelmed at college and can't focus on assignments. My chest tightens and I avoid friends."

# # ---------------------------
# # Display outputs
# # ---------------------------
# if "latest_structured" in st.session_state:
#     st.header("Latest Structured Analysis")
#     structured = st.session_state["latest_structured"]
#     if structured.get("error"):
#         st.warning(structured.get("error_text", "The AI couldn't process this entry."))
#     st.json(structured)

# if "latest_analysis" in st.session_state:
#     st.header("Analysis Summary")
#     analysis = st.session_state["latest_analysis"]
#     if analysis.get("error"):
#         st.warning(analysis.get("error_text", "Analysis could not be generated."))
#     st.json(analysis)
#     df = build_mood_df(load_memory())
#     plot_trends(df)
#     triggers = analysis.get("trigger_counts", {})
#     if triggers:
#         st.subheader("Detected Potential Triggers")
#         st.table(pd.DataFrame(list(triggers.items()), columns=["Trigger", "Count"]).sort_values("Count", ascending=False))

# if "latest_support" in st.session_state:
#     st.header("Support & Practical Steps")
#     support = st.session_state["latest_support"]
#     if support.get("error"):
#         st.warning("Support suggestions could not be generated at this time.")
#     st.markdown(support.get("empathetic_text", "Thank you for sharing your thoughts."))
#     st.subheader("Micro Actions")
#     for action in support.get("micro_actions", []):
#         st.write(f"- {action}")
#     st.subheader("Quick Reframe")
#     st.info(support.get("reframe_prompt", "All feelings are valid."))
#     st.subheader("Breathing Exercise")
#     st.write(support.get("breathing_exercise", "Take a few deep breaths."))

# # Memory overview
# st.markdown("---")
# st.header("Memory Overview")
# mem = load_memory()
# st.write(f"Total entries saved: {len(mem)}")
# if mem:
#     df = build_mood_df(mem)
#     st.dataframe(df[["timestamp", "mood_label", "valence", "arousal"]].sort_values("timestamp", ascending=False).head(20))

# st.caption("This tool is for emotional support and reflection, not a medical diagnostic tool. In crisis, contact local emergency services or a professional.")
