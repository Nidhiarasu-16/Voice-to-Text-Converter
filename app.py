# app.py

import streamlit as st
import tempfile
import requests
import time
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")
st.write("Upload your lecture audio (MP3/WAV) to generate transcript, summarized notes, and quizzes.")

# --- Get AssemblyAI API key ---
API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    st.error(
        "AssemblyAI API key not found! Add it in Streamlit secrets as ASSEMBLYAI_API_KEY."
    )
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
    try:
        with open(audio_path, "rb") as f:
            upload_response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                files={"file": f}
            )
        upload_json = upload_response.json()
        if "upload_url" not in upload_json:
            st.error(f"Upload failed: {upload_json}")
            st.stop()
        audio_url = upload_json["upload_url"]
    except Exception as e:
        st.error(f"Audio upload failed: {e}")
        st.stop()

    # --- Step 2: Request transcription ---
    st.info("Requesting transcription...")
    try:
        transcript_request = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": audio_url}
        )
        transcript_json = transcript_request.json()
        if "id" not in transcript_json:
            st.error(f"Transcription request failed: {transcript_json}")
            st.stop()
        transcript_id = transcript_json["id"]
    except Exception as e:
        st.error(f"Transcription request failed: {e}")
        st.stop()

    # --- Step 3: Poll until transcription is complete ---
    st.info("Transcribing audio... this may take a few moments.")
    while True:
        try:
            status_response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            ).json()
        except Exception as e:
            st.error(f"Error checking transcription status: {e}")
            st.stop()

        if status_response.get("status") == "completed":
            transcript = status_response.get("text", "")
            break
        elif status_response.get("status") == "failed":
            st.error(f"Transcription failed: {status_response}")
            st.stop()
        time.sleep(3)

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 4: Summarize transcript ---
    st.info("Generating summarized study notes...")
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(
            transcript, max_length=200, min_length=80, do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        st.stop()

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 5: Generate Quiz Questions ---
    st.info("Generating quiz questions...")
    try:
        quiz_prompt = f"Create 5 multiple choice questions from this content:\n{transcript}"
        generator = pipeline("text-generation", model="distilgpt2")
        quiz = generator(
            quiz_prompt, max_length=250, do_sample=True, temperature=0.7
        )[0]["generated_text"]
    except Exception as e:
        st.error(f"Error during quiz generation: {e}")
        st.stop()

    st.subheader("❓ Quiz / Flashcards")
    st.write(quiz)

else:
    st.info("Please upload a lecture audio file to start.")
