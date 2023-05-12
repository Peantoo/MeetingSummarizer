import os
from io import BytesIO
from transformers import pipeline
from huggingsound import SpeechRecognitionModel
import requests
from flask import Flask, request, jsonify
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.chains.summarize import load_summarize_chain

app = Flask(__name__)

# Set API Key
APIKEY = os.environ.get("HF_API_KEY")
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# Initialize the Speech Recognition Model
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Summarization
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the file into memory
    audio_data = BytesIO(file.read())

    # Convert the audio to text using the Speech Recognition Model
    transcription_text = model.transcribe(audio_data)
    
    # Generate the summary using the Summarization Chain
    summary_output = chain.run(transcription_text)
    summary_text = summary_output['summary']

    return jsonify({'transcription': transcription_text, 'summary': summary_text}), 200


