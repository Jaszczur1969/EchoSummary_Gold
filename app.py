import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["SPEECHBRAIN_CACHE_LOCAL_ONLY"] = "1"  # opcjonalne: tylko lokalny cache
os.environ["SPEECHBRAIN_LINK_CACHE"] = "copy"

import streamlit as st
import tempfile
import time
from pydub import AudioSegment
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

st.set_page_config(
    page_title="EchoSummary v13",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Styling ----
st.markdown(
    """
    <style>
    .main > .block-container{padding:1.5rem 2rem;}
    .uploader-box {
        border: 1px solid #e6e9ee;
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(248,250,253,0.9));
        box-shadow: 0 4px 18px rgba(16,24,40,0.06);
    }
    .meta { color: #475569; font-size: 14px; }
    .hint { color: #6b7280; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.title("👣 Step 1 - Load file from disk")

    uploaded_file = st.file_uploader(
        "📂 Select an audio or video file",
        type=["mp3", "wav", "m4a", "mp4", "mov", "webm", "mkv"],
        accept_multiple_files=False,
        help="Drag and drop file or click to select."
    )
    st.markdown("**Supported formats:** mp3, wav, m4a, mp4, mov, webm, mkv")

    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        """
            <h3>👣 Step 2 - Provide OpenAI API Key</h3>
            <p style="font-size:13px; color:#6b7280; margin-top:-8px;">
                (required only for transcription and summary generation)
            </p>
        """,
        unsafe_allow_html=True
    )
    api_key = st.text_input(
        "🔑 Enter your API key",
        type="password",
        help="Required only for transcription and summary generation."
    )

    st.markdown("---")
    st.markdown(
        """
            <h3>👣 Step 3 - Provide Hugging Face Token</h3>
            <p style="font-size:13px; color:#6b7280; margin-top:-8px;">
                (required only for transcription and summary generation)
            </p>
        """,
        unsafe_allow_html=True
    )
    hf_token = st.text_input(
        "🔒 Provide Hugging Face token",
        type="password",
        help="Required only for transcription and summary generation."
    )

    # ---- Weryfikacja tokenu Hugging Face ----
    hf_token_valid = False
    if hf_token:
        try:
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            st.success(f"✅ The token is working! Logged in user: {user_info['name']}")
            hf_token_valid = True
        except HfHubHTTPError:
            st.error("❌ The token is invalid or expired. Please check the token and try again.")

# ---- Utils ----
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def plot_waveform(audio_segment, title="📊 Waveform"):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples / np.max(np.abs(samples))
    max_points = 50_000
    if len(samples) > max_points:
        step = len(samples) // max_points
        samples = samples[::step]
    duration_seconds = len(audio_segment) / 1000
    times = np.linspace(0, duration_seconds, num=len(samples))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=samples, mode="lines", line=dict(width=1)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

def transcribe_audio(file_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text

def summarize_text(text, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        "Summarize the text below in no more than 300 words. The summary should be clear and accessible. After the summary, provide a bullet-point list of the key conclusions. Present everything in English.\n\n" + text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content

# ---- Inicjalizacja session_state ----
if "separated_audio_path" not in st.session_state:
    st.session_state.separated_audio_path = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "summary_text" not in st.session_state:
    st.session_state.summary_text = None

# ---- Main area ----
st.markdown("<h1 style='text-align:center;'>🔊 EchoSummary Gold ✒️</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👉 Select a file in the sidebar to get started.")
else:
    # metadata
    st.markdown("<div class='uploader-box'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader(f"{uploaded_file.name}")
        st.markdown(f"<div class='meta'>Size: {sizeof_fmt(len(uploaded_file.getvalue()))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='meta'>Type MIME: {uploaded_file.type or '—'}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1])
    try:
        tfile.write(uploaded_file.getbuffer())
        tfile.flush()
    finally:
        tfile.close()
    temp_path = tfile.name

    # Check type
    video_exts = (".mp4", ".mov", ".webm", ".mkv")
    is_video = uploaded_file.name.lower().endswith(video_exts)

    st.markdown("---")

    # ---- VIDEO ----
    if is_video:
        st.subheader("👣 Step 4 - Video preview")
        st.video(uploaded_file)

        # Separacja audio
        st.subheader("👣 Step 5 - Separate audio from video")
        if st.button("🎵 Click to separate"):
            with st.spinner("Audio extraction..."):
                audio = AudioSegment.from_file(temp_path)  
                audio_file_path = temp_path + ".mp3"
                audio.export(audio_file_path, format="mp3")
                st.session_state.separated_audio_path = audio_file_path
            st.success("✅ Audio separated!")

        # Jeśli audio już odseparowane, pokaż odtwarzacz i waveform
        if st.session_state.separated_audio_path:
            audio_file_path = st.session_state.separated_audio_path
            st.audio(audio_file_path)

            with open(audio_file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download separated audio (mp3)",
                    data=f,
                    file_name=uploaded_file.name.rsplit(".", 1)[0] + ".mp3",
                    mime="audio/mp3"
                )

            audio_segment = AudioSegment.from_file(audio_file_path, format="mp3")
            plot_waveform(audio_segment, title="📊 Waveform of separated audio")

            # ---- Transkrypcja z diarizacją ----
            st.subheader("👣 Step 6 - Transcribe audio (with Speaker Diarization)")
            if st.button("📝 Click to transcribe"):
                if not api_key:
                    st.error("❌ You must enter your API key in the sidebar to transcribe.")
                elif not hf_token_valid:
                    st.error("❌ You must provide a valid Hugging Face token to perform diarization.")
                else:
                    with st.spinner("Speaker diarization and transcription are in progress..."):
                        from pyannote.audio import Pipeline
                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
                        diarization = pipeline(audio_file_path)
                        transcript = ""
                        for segment, _, speaker in diarization.itertracks(yield_label=True):
                            start_time = segment.start
                            end_time = segment.end
                            speaker_label = f"Speaker {speaker}"
                            transcript += f"[{speaker_label} ({start_time:.2f}s - {end_time:.2f}s)]:\n"
                            segment_audio = AudioSegment.from_file(audio_file_path, format="mp3")[int(start_time*1000):int(end_time*1000)]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_segment:
                                segment_audio.export(tmp_segment.name, format="mp3")
                                segment_transcript = transcribe_audio(tmp_segment.name, api_key)
                            transcript += segment_transcript + "\n\n"
                        st.session_state.transcript_text = transcript
                    st.success("✅ Transcription with diarization completed!")

            if st.session_state.transcript_text:
                edited_text = st.text_area(
                    "✏️ Transcript (you can edit)",
                    value=st.session_state.transcript_text,
                    height=300
                )
                if edited_text:
                    st.download_button(
                        label="⬇️ Download transcript (TXT)",
                        data=edited_text,
                        file_name=uploaded_file.name.rsplit('.',1)[0] + ".txt",
                        mime="text/plain"
                    )

                # ---- Uruchamianie tworzenia streszczenia ----
                st.subheader("👣 Step 7 - Create a summary (max 300 words)")
                if st.button("📖 Click to create a summary"):
                    if not api_key:
                        st.error("❌ You must provide an API key to generate a summary.")
                    else:
                        with st.spinner("Creating a summary..."):
                            summary = summarize_text(edited_text, api_key)
                            st.session_state.summary_text = summary
                        st.success("✅ The summary is ready!")

                if st.session_state.summary_text:
                    st.text_area("📖 Summary (max 300 words)", value=st.session_state.summary_text, height=250)
                    st.download_button(
                        label="⬇️ Download the summary (TXT)",
                        data=st.session_state.summary_text,
                        file_name=uploaded_file.name.rsplit('.',1)[0] + "_summary.txt",
                        mime="text/plain"
                    )

    # ---- AUDIO ----
    else:
        st.subheader("👣 Step 4 - Audio preview")
        st.audio(uploaded_file)

        # Waveform
        audio_segment = AudioSegment.from_file(uploaded_file, format=uploaded_file.name.split(".")[-1])
        plot_waveform(audio_segment)

        audio_file_path = temp_path  

        # ---- Transkrypcja z diarizacją dla zwykłego audio ----

        if st.button("📝 Transcribe audio (Speaker Diarization)"):
            if not api_key:
                st.error("❌ You must enter your API key in the sidebar to transcribe.")
            elif not hf_token_valid:
                st.error("❌ You must provide a valid Hugging Face token to perform diarization.")
            else:
                with st.spinner("Speaker diarization and transcription are in progress..."):
                    from pyannote.audio import Pipeline
                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
                    diarization = pipeline(audio_file_path)
                    transcript = ""
                    for segment, _, speaker in diarization.itertracks(yield_label=True):
                        start_time = segment.start
                        end_time = segment.end
                        speaker_label = f"Speaker {speaker}"
                        transcript += f"[{speaker_label} ({start_time:.2f}s - {end_time:.2f}s)]:\n"
                        segment_audio = AudioSegment.from_file(audio_file_path, format=uploaded_file.name.split(".")[-1])[int(start_time*1000):int(end_time*1000)]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_segment:
                            segment_audio.export(tmp_segment.name, format="mp3")
                            segment_transcript = transcribe_audio(tmp_segment.name, api_key)
                        transcript += segment_transcript + "\n\n"
                    st.session_state.transcript_text = transcript
                st.success("✅ Transcription with diarization completed!")

        if st.session_state.transcript_text:
            edited_text = st.text_area(
                "✏️ Transcript (you can edit)",
                value=st.session_state.transcript_text,
                height=300
            )
            if edited_text:
                st.download_button(
                    label="⬇️ Download transcript (TXT)",
                    data=edited_text,
                    file_name=uploaded_file.name.rsplit('.',1)[0] + ".txt",
                    mime="text/plain"
                )

            # ---- Podsumowanie ----
            if st.button("📖 Create a summary (max 300 words)"):
                if not api_key:
                    st.error("❌ You must provide an API key to generate a summary.")
                else:
                    with st.spinner("Creating a summary..."):
                        summary = summarize_text(edited_text, api_key)
                        st.session_state.summary_text = summary
                    st.success("✅ The summary is ready!")

            if st.session_state.summary_text:
                st.text_area("📖 Summary (max 300 words)", value=st.session_state.summary_text, height=250)
                st.download_button(
                    label="⬇️ Download the summary (TXT)",
                    data=st.session_state.summary_text,
                    file_name=uploaded_file.name.rsplit('.',1)[0] + "_summary.txt",
                    mime="text/plain"
                )

    st.markdown("---")