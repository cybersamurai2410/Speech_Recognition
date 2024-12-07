import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset
from transformers import pipeline

# Initialize pipelines for speech recognition and tts models
asr = pipeline(task="automatic-speech-recognition", model="distil-whisper/distil-small.en")
narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

# Speech-to-Text Function
def transcribe_speech(filepath):
    if filepath is None:
        return "No audio found. Please retry."
    output = asr(filepath)
    return output["text"]

# Long-Form Audio Transcription
def transcribe_long_form(filepath):
    if filepath is None:
        return "No audio found. Please retry."
    # Load and preprocess audio
    audio, sampling_rate = sf.read(filepath)
    audio_transposed = np.transpose(audio)
    audio_mono = librosa.to_mono(audio_transposed)
    audio_16KHz = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=16000)
    # Transcribe using ASR pipeline
    chunks = asr(audio_16KHz, chunk_length_s=30, batch_size=4, return_timestamps=True)["chunks"]
    # Combine all transcriptions
    return "\n".join([chunk["text"] for chunk in chunks])

# Text-to-Speech Function
def text_to_speech(text):
    if not text.strip():
        return "No text provided. Please enter text to synthesize."
    narrated_text = narrator(text)
    audio_array = narrated_text["audio"][0].flatten()  # Flatten the 2D array to 1D
    sampling_rate = narrated_text["sampling_rate"]  # Get sampling rate
    return sampling_rate, audio_array

# Sample Dataset Access Function
def get_dataset_sample(idx):
    dataset = load_dataset("librispeech_asr", split="train.clean.100", streaming=True, trust_remote_code=True)
    # example = next(iter(dataset))  
    dataset_head = list(dataset.take(5)) 
    sample = dataset_head[idx]
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    transcription = sample["text"]
    return (audio_array, sampling_rate), transcription
  
