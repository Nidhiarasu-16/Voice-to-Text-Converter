# app.py

import streamlit as st
import tempfile
import requests
import time
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")
st.write("Upload your lecture audio (MP3/WAV) to generate transcript, summarized notes, and quizzes.")

# --- Get AssemblyAI key ---
API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    st.error("AssemblyAI API key not found! Add it in Streamlit secrets as ASSEMBLYAI_API_KEY.")
    st.stop()

headers = {"authorization": API_KEY}

# --- Upload audio ---
st.subheader("Upload Lecture Audio (MP3/WAV)")
audio_file = st.file_uploader("Choose an audio file", type=["mp3","wav"])

if audio_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    st.audio(audio_file, format="audio/wav")

    # --- Step 1: Upload to AssemblyAI ---
    st.info("Uploading audio for transcription...")
    upload_response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        files={"file": open(audio_path, "rb")}
    )
    audio_url = upload_response.json()["upload_url"]

    # --- Step 2: Request transcription ---
    st.info("Transcribing audio... this may take a few moments.")
    transcript_request = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": audio_url}
    )
    transcript_id = transcript_request.json()["id"]

    # --- Step 3: Poll until transcription is complete ---
    while True:
        status_response = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers
        ).json()
        if status_response["status"] == "completed":
            transcript = status_response["text"]
            break
        elif status_response["status"] == "failed":
            st.error("Transcription failed.")
            st.stop()
        time.sleep(3)

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 4: Summarize transcript ---
    st.info("Generating summarized study notes...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=200, min_length=80, do_sample=False)[0]["summary_text"]

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 5: Generate Quiz ---
    st.info("Generating quiz questions...")
    quiz_prompt = f"Create 5 multiple choice questions from this content:\n{transcript}"
    generator = pipeline("text-generation", model="distilgpt2")
    quiz = generator(quiz_prompt, max_length=250, do_sample=True, temperature=0.7)[0]["generated_text"]

    st.subheader("❓ Quiz / Flashcards")
    st.write(quiz)

else:
    st.info("Please upload a lecture audio file to start.")
