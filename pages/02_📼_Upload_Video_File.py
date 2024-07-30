import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
import mlx_whisper
import ffmpeg
import requests
from typing import List, Dict, Any
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit page config
st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")

# Define constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODEL_NAME = "mlx-community/whisper-small"
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

def process_audio(input_file: str, output_file: str) -> None:
    try:
        audio = ffmpeg.input(input_file)
        audio = ffmpeg.output(audio, output_file, acodec="pcm_s16le", ac=1, ar="16k")
        ffmpeg.run(audio, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

def generate_subtitled_video(video: str, audio: str, transcript: str, output: str) -> None:
    try:
        video_file = ffmpeg.input(video)
        audio_file = ffmpeg.input(audio)
        ffmpeg.concat(
            video_file.filter("subtitles", transcript),
            audio_file,
            v=1,
            a=1
        ).output(output).run(quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error while generating subtitled video: {e.stderr.decode()}")
        raise

def load_whisper_model(model_size: str = "small") -> mlx_whisper.Whisper:
    try:
        model = mlx_whisper.load_model(model_size)
        return model
    except Exception as e:
        st.error(f"Failed to load MLX Whisper model: {e}")
        raise

def inference(model: mlx_whisper.Whisper, audio_path: str, task: str) -> Dict[str, Any]:
    try:
        options = {"task": task, "best_of": 5}
        result = mlx_whisper.transcribe(audio_path, model=model, **options)
        return result
    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise

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
                
                # Process audio
                audio_path = str(SAVE_DIR / "output.wav")
                process_audio(input_path, audio_path)
                
                # Load MLX Whisper model
                model = load_whisper_model(MODEL_NAME)
                
                # Perform inference
                results = inference(model, audio_path, task.lower())
                
                # Display results
                col3, col4 = st.columns(2)
                with col3:
                    st.video(input_file)
                
                # Write subtitles
                vtt_path = str(SAVE_DIR / "transcript.vtt")
                srt_path = str(SAVE_DIR / "transcript.srt")
                write_subtitles(results["segments"], "vtt", vtt_path)
                write_subtitles(results["segments"], "srt", srt_path)
                
                # Generate subtitled video
                output_video_path = str(SAVE_DIR / "final.mp4")
                generate_subtitled_video(input_path, audio_path, srt_path, output_video_path)
                
                with col4:
                    st.video(output_video_path)
                    st.success(f"{task} completed successfully!")
                
                # Create zip file with all outputs
                zip_path = str(SAVE_DIR / "transcripts_and_video.zip")
                with ZipFile(zip_path, "w") as zipf:
                    for file in [vtt_path, srt_path, output_video_path]:
                        zipf.write(file, os.path.basename(file))
                
                # Create download link
                st.markdown(create_download_link(zip_path, "Download Transcripts and Video"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.exception("Error in main processing loop")

if __name__ == "__main__":
    main()