# app.py

import streamlit as st
import tempfile
import openai
from transformers import pipeline

# --- Page setup ---
st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")
st.write(
    "Upload your lecture audio (MP3/WAV) to generate transcript, summarized notes, and quizzes."
)

# --- OpenAI API key from Streamlit secrets ---
openai.api_key = st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.error(
        "OpenAI API key not found! Please add it in Streamlit Cloud secrets as OPENAI_API_KEY."
    )
    st.stop()

# --- Upload audio ---
st.subheader("Upload Lecture Audio (MP3/WAV)")
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if audio_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    # Play uploaded audio
    st.audio(audio_file, format="audio/wav")

    # --- Step 1: Transcribe audio using OpenAI Whisper API ---
    st.info("Transcribing audio... Please wait.")
    try:
        with open(audio_path, "rb") as f:
            response = openai.audio.transcriptions.create(
                file=f,
                model="whisper-1"
            )
        transcript = response["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.stop()

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 2: Summarize transcript ---
    st.info("Generating summarized study notes...")
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(
            transcript,
            max_length=200,
            min_length=80,
            do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        st.stop()

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 3: Generate Quiz Questions ---
    st.info("Generating quiz questions...")
    try:
        quiz_prompt = f"Create 5 multiple choice questions from this content:\n{transcript}"
        generator = pipeline("text-generation", model="gpt2")
        quiz = generator(
            quiz_prompt,
            max_length=250,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]
    except Exception as e:
        st.error(f"Error during quiz generation: {e}")
        st.stop()

    st.subheader("❓ Quiz / Flashcards")
    st.write(quiz)

else:
    st.info("Please upload a lecture audio file to start.")
