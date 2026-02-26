import streamlit as st
import tempfile
import os
import wave
import json
import zipfile
from vosk import Model, KaldiRecognizer
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🔊 ➜ 📝")
st.write(
    "Upload your lecture audio (MP3/WAV) to generate transcript, summarized notes, and quizzes."
)

# --- Step 0: Download Vosk model if not present ---
MODEL_PATH = "model"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading Vosk speech recognition model (~50MB)... This may take a minute.")
    import wget

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    wget.download(url, "vosk-model.zip")
    with zipfile.ZipFile("vosk-model.zip", "r") as zip_ref:
        zip_ref.extractall(MODEL_PATH)
    st.success("Model downloaded!")

# Load the Vosk model
st.info("Loading speech recognition model...")
try:
    vosk_model = Model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load Vosk model. Error: {e}")
    st.stop()

# --- Step 1: Upload audio ---
st.subheader("Upload Lecture Audio (WAV recommended)")
audio_file = st.file_uploader("Choose an audio file", type=["wav","mp3"])

if audio_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tfile.write(audio_file.read())
    audio_path = tfile.name

    st.audio(audio_file, format="audio/wav")

    # --- Step 2: Transcribe audio with Vosk ---
    st.info("Transcribing audio with Vosk...")
    try:
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        transcript = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += " " + res.get("text", "")
        final_res = json.loads(rec.FinalResult())
        transcript += " " + final_res.get("text", "")
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.stop()

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 3: Summarize transcript ---
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

    # --- Step 4: Generate quiz questions ---
    st.info("Generating quiz questions...")
    try:
        quiz_prompt = f"Create 5 multiple choice questions from this content:\n{transcript}"
        generator = pipeline("text-generation", model="gpt2")
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
