import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
import mlx_whisper
import requests
from typing import List, Dict, Any
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
import subprocess
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit page config
st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")

# Define constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODEL_NAME = "mlx-community/whisper-large-v3-mlx"
APP_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_DIR = APP_DIR / "local_video"
LOCAL_DIR.mkdir(exist_ok=True)
SAVE_DIR = LOCAL_DIR / "output"
SAVE_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_lottie_url(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logging.error(f"Failed to load Lottie animation: {e}")
        return None

def load_whisper_model(model_name: str = "mlx-community/whisper-small"):
    try:
        model = mlx_whisper.load_models.load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load MLX Whisper model: {e}")
        raise

def prepare_audio(audio_path: str) -> mx.array:
    command = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    return mx.array(audio_array)

def process_audio(model, audio: mx.array, task: str) -> Dict[str, Any]:
    options = {"task": task}
    results = mlx_whisper.transcribe(audio, model=model, **options)
    return results

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
            for segment in segments:
                f.write(f"{segment['start']:.3f} --> {segment['end']:.3f}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        elif format == "srt":
            for i, segment in enumerate(segments, start=1):
                f.write(f"{i}\n")
                start = f"{int(segment['start'] // 3600):02d}:{int(segment['start'] % 3600 // 60):02d}:{segment['start'] % 60:06.3f}"
                end = f"{int(segment['end'] // 3600):02d}:{int(segment['end'] % 3600 // 60):02d}:{segment['end'] % 60:06.3f}"
                f.write(f"{start.replace('.', ',')} --> {end.replace('.', ',')}\n")
                f.write(f"{segment['text'].strip()}\n\n")

def create_download_link(file_path: str, link_text: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

def main():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        lottie = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_HjK9Ol.json")
        if lottie:
            st_lottie(lottie)
    
    with col2:
        st.write("""
        ## Auto Subtitled Video Generator 
        ##### Upload a video file and get a video with subtitles.
        ###### ➠ If you want to transcribe the video in its original language, select the task as "Transcribe"
        ###### ➠ If you want to translate the subtitles to English, select the task as "Translate"
        """)
    
    input_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    task = st.selectbox("Select Task", ["Transcribe", "Translate"], index=0)
    
    if input_file and st.button(task):
        with st.spinner(f"{task}ing the video..."):
            try:
                # Save uploaded file
                input_path = str(SAVE_DIR / "input.mp4")
                with open(input_path, "wb") as f:
                    f.write(input_file.read())
                
                # Load MLX Whisper model
                model = load_whisper_model(MODEL_NAME)
                
                # Prepare audio and perform inference
                audio = prepare_audio(input_path)
                results = process_audio(model, audio, task.lower())
                
                # Display results
                col3, col4 = st.columns(2)
                with col3:
                    st.video(input_file)
                
                # Write subtitles
                vtt_path = str(SAVE_DIR / "transcript.vtt")
                srt_path = str(SAVE_DIR / "transcript.srt")
                write_subtitles(results["segments"], "vtt", vtt_path)
                write_subtitles(results["segments"], "srt", srt_path)
                
                with col4:
                    st.text_area("Transcription", results["text"], height=300)
                    st.success(f"{task} completed successfully!")
                
                # Create zip file with outputs
                zip_path = str(SAVE_DIR / "transcripts.zip")
                with ZipFile(zip_path, "w") as zipf:
                    for file in [vtt_path, srt_path]:
                        zipf.write(file, os.path.basename(file))
                
                # Create download link
                st.markdown(create_download_link(zip_path, "Download Transcripts"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.exception("Error in main processing loop")

if __name__ == "__main__":
    main()