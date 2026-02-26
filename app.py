import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
from transformers import pipeline
import numpy as np
import tempfile
import av
import soundfile as sf

st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")
st.title("Lecture Voice-to-Notes Generator 🎓")
st.write("Record your lecture live and generate summarized notes & quizzes instantly!")

# --- Step 1: Live Audio Recording ---
st.subheader("🎤 Record your lecture")
st.write("Press 'Start' to record. When finished, press 'Stop'.")

audio_file_path = st.text_input("Audio file path will appear here after recording", "")

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to numpy
        frame_ndarray = frame.to_ndarray()
        # Take one channel only
        if frame_ndarray.ndim > 1:
            frame_ndarray = frame_ndarray[0]
        self.audio_frames.append(frame_ndarray)
        return frame

webrtc_ctx = webrtc_streamer(
    key="lecture-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True},
    async_processing=True,
)

# Button to save recorded audio
if webrtc_ctx.state.playing:
    st.info("Recording in progress...")

if st.button("Save Recording"):
    if webrtc_ctx.audio_processor:
        audio_data = np.hstack(webrtc_ctx.audio_processor.audio_frames)
        # Save as WAV
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio_data, samplerate=48000)
        audio_file_path = temp_wav.name
        st.success(f"Recording saved: {audio_file_path}")
        st.text_input("Audio file path will appear here after recording", value=audio_file_path)

# --- Step 2: Transcription ---
if audio_file_path:
    st.info("Transcribing audio...")
    model = whisper.load_model("base")  # use "tiny" for faster transcription
    result = model.transcribe(audio_file_path)
    transcript = result["text"]

    st.subheader("📄 Transcript")
    st.write(transcript)

    # --- Step 3: Summarization ---
    st.info("Generating summarized study notes...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=250, min_length=100, do_sample=False)[0]['summary_text']

    st.subheader("📝 Study Notes")
    st.write(summary)

    # --- Step 4: Quiz Generation ---
    st.info("Generating simple quiz questions...")
    quiz_prompt = f"Create 5 multiple-choice questions from the following lecture:\n{transcript}"
    generator = pipeline("text-generation", model="gpt2")
    quiz = generator(quiz_prompt, max_length=300, do_sample=True, temperature=0.7)[0]['generated_text']

    st.subheader("❓ Quiz / Flashcards")

    st.write(quiz)
