import gradio as gr
from ts_utilites import transcribe_speech, transcribe_long_form, text_to_speech

# Gradio App
app = gr.TabbedInterface(
    [
        # Transcribe Speech
        gr.Interface(
            fn=transcribe_speech,
            inputs=gr.Audio(type="filepath"),
            outputs=gr.Textbox(label="Transcription", lines=5),
            title="Transcribe Speech",
            allow_flagging="never",
        ),
        # Long-Form Transcription
        gr.Interface(
            fn=transcribe_long_form,
            inputs=gr.Audio(type="filepath"),
            outputs=gr.Textbox(label="Transcription", lines=10),
            title="Long-Form Transcription",
            allow_flagging="never",
        ),
        # Text-to-Speech
        gr.Interface(
            fn=text_to_speech,
            inputs=gr.Textbox(label="Enter Text", placeholder="Type your text here...", lines=5),
            outputs=gr.Audio(label="Generated Speech"),
            title="Text-to-Speech",
            allow_flagging="never",
        )
    ],
    ["Transcribe Microphone", "Transcribe Audio File", "Long-Form Transcription", "Dataset Samples", "Text-to-Speech"]
)
app.launch()
