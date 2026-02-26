import streamlit as st
import tempfile
import speech_recognition as sr
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")
st.write("Upload your lecture audio (WAV) to generate transcript, summarized notes, and quizzes.")

# --- Upload audio ---
st.subheader("Upload Lecture Audio (WAV only)")
audio_file = st.file_uploader("Choose an audio file", type=["wav"])

if audio_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    st.audio(audio_file, format="audio/wav")

    # --- Step 1: Transcribe audio using CMU Sphinx ---
    st.info("Transcribing audio with CMU Sphinx (offline)...")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcript = recognizer.recognize_sphinx(audio)
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
            transcript, max_length=200, min_length=80, do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        st.stop()

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 3: Generate quiz questions ---
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
    st.info("Please upload a lecture WAV audio file to start.")
