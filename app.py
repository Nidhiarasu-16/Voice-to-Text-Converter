import streamlit as st
import tempfile
import openai
from transformers import pipeline

st.set_page_config(page_title="Voice-to-Notes Generator", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")

# Get API Key from environment
openai.api_key = st.secrets.get("OPENAI_API_KEY")

st.subheader("Upload Lecture Audio (MP3/WAV)")
audio_file = st.file_uploader("Choose an audio file", type=["mp3","wav"])

if audio_file:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    st.audio(audio_file, format="audio/wav")

    # --- Transcription via OpenAI Whisper API ---
    st.info("Transcribing audio... Please wait.")
    with open(audio_path, "rb") as f:
        response = openai.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        )

    transcript = response["text"]

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Summarization ---
    st.info("Generating study notes...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(
        transcript, max_length=200, min_length=80, do_sample=False
    )[0]["summary_text"]

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Quiz Generation ---
    st.info("Generating quiz questions...")
    quiz_prompt = f"Create 5 multiple choice questions from this content:\n{transcript}"

    generator = pipeline("text-generation", model="gpt2")
    quiz = generator(
        quiz_prompt, max_length=250, do_sample=True, temperature=0.7
    )[0]["generated_text"]

    st.subheader("❓ Quiz Questions")
    st.write(quiz)

else:
    st.info("Please upload an audio file to start.")
