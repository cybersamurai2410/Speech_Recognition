import gradio as gr
import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset
from transformers import pipeline

# Load Dataset
dataset = load_dataset("librispeech_asr", split="train.clean.100", streaming=True, trust_remote_code=True)
example = next(iter(dataset))  # Get the first example
dataset_head = list(dataset.take(5))  # Get the first 5 examples

# Initialize Pipelines
asr = pipeline(task="automatic-speech-recognition", model="./models/distil-whisper/distil-small.en")
narrator = pipeline("text-to-speech", model="./models/kakao-enterprise/vits-ljs")

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
    return narrated_text["audio"][0]

# Sample Dataset Access Function
def get_dataset_sample(idx):
    """
    Fetches a sample audio and transcription from the dataset.
    Args:
        idx (int): Index of the sample to fetch.
    Returns:
        tuple: Audio array and transcription.
    """
    sample = dataset_head[idx]
    transcription = sample["text"]
    return sample["audio"]["array"], transcription

# Gradio App
with gr.Blocks() as demo:
    with gr.TabbedInterface():
        # Transcribe Microphone
        with gr.TabItem("Transcribe Microphone"):
            gr.Interface(
                fn=transcribe_speech,
                inputs=gr.Audio(source="microphone", type="filepath"),
                outputs=gr.Textbox(label="Transcription", lines=5),
            )

        # Transcribe Audio File
        with gr.TabItem("Transcribe Audio File"):
            gr.Interface(
                fn=transcribe_speech,
                inputs=gr.Audio(source="upload", type="filepath"),
                outputs=gr.Textbox(label="Transcription", lines=5),
            )

        # Long-Form Transcription
        with gr.TabItem("Long-Form Transcription"):
            gr.Interface(
                fn=transcribe_long_form,
                inputs=gr.Audio(source="upload", type="filepath"),
                outputs=gr.Textbox(label="Transcription", lines=10),
            )

        # Dataset Sample Access
        with gr.TabItem("Dataset Samples"):
            sample_idx = gr.Slider(0, len(dataset_head) - 1, step=1, label="Sample Index")
            audio_output = gr.Audio()
            transcription_output = gr.Textbox(label="Transcription")
            sample_button = gr.Button("Get Sample")
            sample_button.click(
                fn=get_dataset_sample,
                inputs=[sample_idx],
                outputs=[audio_output, transcription_output],
            )

        # Text-to-Speech
        with gr.TabItem("Text-to-Speech"):
            gr.Interface(
                fn=text_to_speech,
                inputs=gr.Textbox(label="Enter Text", placeholder="Type your text here...", lines=5),
                outputs=gr.Audio(label="Generated Speech"),
            )

if __name__ == "__main__":
    demo.launch(share=True)
