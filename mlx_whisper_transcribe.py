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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit page config
st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")

# Define constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODELS = {
    "Tiny (Q4)": "mlx-community/whisper-tiny-mlx-q4",
    "Large v3": "mlx-community/whisper-large-v3-mlx",
    "Small English (Q4)": "mlx-community/whisper-small.en-mlx-q4",
    "Small (FP32)": "mlx-community/whisper-small-mlx-fp32",
    "Distil Large v3": "mlx-community/distil-whisper-large-v3"  # New model added here
}
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

def process_audio(model_path: str, audio: mx.array, task: str) -> Dict[str, Any]:
    logging.info(f"Processing audio with model: {model_path}, task: {task}")
    
    try:
        if task.lower() == "transcribe":
            results = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_path,
                fp16=False,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        logging.info(f"{task.capitalize()} completed successfully")
        return results
    except Exception as e:
        logging.error(f"Unexpected error in mlx_whisper.{task}: {e}")
        raise

def split_long_caption(text: str, max_chars: int = 42) -> List[str]:
    sentences = re.split('(?<=[.!?]) +', text)
    lines = []
    for sentence in sentences:
        words = sentence.split()
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + (1 if current_line else 0) <= max_chars:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(" ".join(current_line))
    return lines

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments, start=1):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            
            # Split long captions
            caption_lines = split_long_caption(text)
            
            # Calculate duration for each line
            total_duration = max(end - start, 0.001)  # Ensure minimum duration
            words_per_minute = 160
            chars_per_word = 5  # Approximate average
            chars_per_minute = words_per_minute * chars_per_word
            chars_per_second = chars_per_minute / 60
            
            min_duration = 1.0  # Minimum duration in seconds
            max_duration = 7.0  # Maximum duration in seconds
            
            for j in range(0, len(caption_lines), 2):
                two_lines = caption_lines[j:j+2]
                line_text = "\n".join(two_lines)
                line_chars = sum(len(line) for line in two_lines)
                
                line_duration = max(min(line_chars / chars_per_second, max_duration), min_duration)
                line_start = start + (j / len(caption_lines)) * total_duration
                line_end = min(line_start + line_duration, end)
                
                if format == "vtt":
                    f.write(f"{line_start:.3f} --> {line_end:.3f}\n")
                    f.write(f"{line_text}\n\n")
                elif format == "srt":
                    f.write(f"{i}.{j//2+1}\n")
                    start_time = f"{int(line_start // 3600):02d}:{int(line_start % 3600 // 60):02d}:{line_start % 60:06.3f}"
                    end_time = f"{int(line_end // 3600):02d}:{int(line_end % 3600 // 60):02d}:{line_end % 60:06.3f}"
                    f.write(f"{start_time.replace('.', ',')} --> {end_time.replace('.', ',')}\n")
                    f.write(f"{line_text}\n\n")

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
        st.markdown("""
            ## Apple MLX Powered Video Transcription

            Upload your video and get:
            - Accurate transcripts (SRT/VTT files)
            - Lightning-fast processing

            🎙️ Transcribe: Capture spoken words in the original language
        """)
    
    input_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    
    # Add model selection dropdown without tooltip
    selected_model = st.selectbox(
        "Select Whisper Model",
        list(MODELS.keys()),
        index=4
    )
    MODEL_NAME = MODELS[selected_model]
    
    # Add information about the Distil Large v3 model
    if selected_model == "Distil Large v3":
        st.info("""
        **Distil Large v3 Model**
        
        This new model offers significant performance improvements:
        - Runs approximately 40 times faster than real-time on M1 Max chips
        - Can transcribe 12 minutes of audio in just 18 seconds
        - Provides a great balance between speed and accuracy
        
        Ideal for processing longer videos or when you need quick results without sacrificing too much accuracy.
        """)
    
    if input_file and st.button("Transcribe"):
        with st.spinner(f"Transcribing the video using {selected_model} model..."):
            try:
                # Save uploaded file
                input_path = str(SAVE_DIR / "input.mp4")
                with open(input_path, "wb") as f:
                    f.write(input_file.read())
                
                # Prepare audio
                audio = prepare_audio(input_path)
                
                # Process audio
                results = process_audio(MODEL_NAME, audio, "transcribe")
                
                # Write subtitles
                vtt_path = str(SAVE_DIR / "transcript.vtt")
                srt_path = str(SAVE_DIR / "transcript.srt")
                try:
                    write_subtitles(results["segments"], "vtt", vtt_path)
                    write_subtitles(results["segments"], "srt", srt_path)
                except Exception as subtitle_error:
                    st.error(f"Error writing subtitles: {str(subtitle_error)}")
                    logging.exception("Error writing subtitles")
                else:
                    # Create zip file with outputs
                    zip_path = str(SAVE_DIR / "transcripts.zip")
                    with ZipFile(zip_path, "w") as zipf:
                        for file in [vtt_path, srt_path]:
                            zipf.write(file, os.path.basename(file))
                    
                    # Create download link
                    st.markdown(create_download_link(zip_path, "Download Transcripts"), unsafe_allow_html=True)
                
                # Display results
                col3, col4 = st.columns(2)
                with col3:
                    st.video(input_file)
                
                with col4:
                    st.text_area("Transcription", results["text"], height=300)
                    st.success(f"Transcription completed successfully using {selected_model} model!")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.exception("Error in main processing loop")

if __name__ == "__main__":
    main()