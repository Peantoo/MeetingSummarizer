import os
from io import BytesIO
from transformers import pipeline
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

# Speech-to-text
API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
headers = {"Authorization": APIKEY}

def query(audio_data):
    response = requests.post(API_URL, headers=headers, data=audio_data)
    return response.json()

# Summarization
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")


#summarizer = LLMChain(llm=llm, prompt=PromptTemplate("summarize {text}"), output_key="summary")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the file into memory
    audio_data = BytesIO(file.read())

    # Placeholder block for Speech-to-Text code
    transcription_output = query(audio_data.getvalue())
    transcription_text = transcription_output['text']
    
    summary_output = chain.run(transcription_text)
    #summary_output = summarizer.run(text=transcription_text)
    summary_text = summary_output['summary']

    return jsonify({'transcription': transcription_text, 'summary': summary_text}), 200

if __name__ == '__main__':
    try:
        # Run Streamlit interface
        streamlit_interface()
    except Exception as e:
        print(f"Error: {e}")
