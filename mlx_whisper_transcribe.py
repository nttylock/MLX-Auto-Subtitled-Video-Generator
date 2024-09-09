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
                verbose=True,
                word_timestamps=True  # Enable word-level timestamps
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        logging.info(f"{task.capitalize()} completed successfully")
        return results
    except Exception as e:
        logging.error(f"Unexpected error in mlx_whisper.{task}: {e}")
        raise

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
        
        subtitle_count = 1
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            text = ' '.join(word['word'] for word in words)
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            
            # Split into lines of maximum 42 characters
            lines = []
            current_line = []
            current_length = 0
            for word in text.split():
                if current_length + len(word) + 1 > 42:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            
            # Group lines into subtitles with a maximum of 2 lines each
            for i in range(0, len(lines), 2):
                subtitle_lines = lines[i:i+2]
                subtitle_text = '\n'.join(subtitle_lines)
                
                start_index = sum(len(line.split()) for line in lines[:i])
                end_index = min(sum(len(line.split()) for line in lines[:i+2]), len(words))
                
                start_word = words[start_index]
                end_word = words[end_index - 1]
                
                start_time = start_word['start']
                end_time = end_word['end']
                
                # Ensure minimum duration
                duration = end_time - start_time
                min_duration = max(len(subtitle_text) / 21, 1.5)  # At least 1.5 seconds or 21 characters per second
                if duration < min_duration:
                    end_time = start_time + min_duration
                
                if format == "srt":
                    f.write(f"{subtitle_count}\n")
                    f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                elif format == "vtt":
                    f.write(f"{format_timestamp(start_time, vtt=True)} --> {format_timestamp(end_time, vtt=True)}\n")
                    f.write(f"{subtitle_text}\n\n")
                
                subtitle_count += 1
            
            # Check for potential data loss
            processed_words = ' '.join(lines).split()
            original_words = ' '.join(word['word'] for word in words).split()
            if len(processed_words) != len(original_words):
                logging.warning(f"Potential data loss detected in segment {segment.get('id', 'unknown')}")
                logging.warning(f"Original: {' '.join(original_words)}")
                logging.warning(f"Processed: {' '.join(processed_words)}")

    # After processing all segments
    original_text = ' '.join(seg['text'] for seg in segments)
    final_text = ' '.join(line.strip() for line in open(output_file, 'r', encoding='utf-8').readlines() if line.strip() and not line[0].isdigit() and '-->' not in line)
    if original_text != final_text:
        logging.warning("Potential data loss or word order change detected in final output")

def write_text_transcription(segments: List[Dict[str, Any]], output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment['text']
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            f.write(text + "\n")

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if vtt:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    else:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace('.', ',')

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
                
                # Write subtitles and text transcription
                vtt_path = str(SAVE_DIR / "transcript.vtt")
                srt_path = str(SAVE_DIR / "transcript.srt")
                txt_path = str(SAVE_DIR / "transcript.txt")
                try:
                    write_subtitles(results["segments"], "vtt", vtt_path)
                    write_subtitles(results["segments"], "srt", srt_path)
                    write_text_transcription(results["segments"], txt_path)
                except Exception as subtitle_error:
                    st.error(f"Error writing subtitles or transcription: {str(subtitle_error)}")
                    logging.exception("Error writing subtitles or transcription")
                else:
                    # Create zip file with outputs
                    zip_path = str(SAVE_DIR / "transcripts.zip")
                    with ZipFile(zip_path, "w") as zipf:
                        for file in [vtt_path, srt_path, txt_path]:
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