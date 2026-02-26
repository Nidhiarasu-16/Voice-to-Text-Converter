import streamlit as st
import tempfile
import whisper
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🎓")
st.write(
    "Upload your lecture audio (MP3/WAV) to generate transcript, summarized notes, and quizzes."
)

# --- Step 1: Upload Audio ---
st.subheader("Upload your lecture audio")
audio_file = st.file_uploader("MP3/WAV only", type=["mp3", "wav"])

if audio_file:
    # Save uploaded audio temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    st.audio(audio_file, format='audio/wav')

    # --- Step 2: Transcribe Audio using Whisper ---
    st.info("Transcribing audio... (this may take a few seconds)")
    model = whisper.load_model("tiny")  # "tiny" is fast and works on Cloud
    result = model.transcribe(audio_path)
    transcript = result["text"]

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 3: Summarize Transcript ---
    st.info("Generating summarized study notes...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(
        transcript, max_length=250, min_length=100, do_sample=False
    )[0]["summary_text"]

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 4: Generate Quiz Questions ---
    st.info("Generating simple quiz questions...")
    quiz_prompt = f"Create 5 multiple-choice questions from the following lecture:\n{transcript}"

    generator = pipeline("text-generation", model="gpt2")
    quiz = generator(
        quiz_prompt, max_length=300, do_sample=True, temperature=0.7
    )[0]["generated_text"]

    st.subheader("❓ Quiz / Flashcards")
    st.write(quiz)

else:
    st.info("Please upload a lecture audio file to begin.")
