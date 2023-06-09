import os
import base64
import tempfile
from flask import Flask, request, render_template, jsonify
import speech_recognition as sr
import openai
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

app = Flask(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")
USERNAME = os.environ.get("APP_USERNAME")
PASSWORD = os.environ.get("APP_PASSWORD")
TOKENS = int(os.environ.get("APP_TOKENS"))
OPENAI_ENGINE = os.environ.get("OPENAI_ENGINE", "gpt-3.5-turbo")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
            return render_template("main.html")
        else:
            return "Invalid credentials. Please try again."
    return render_template("login.html")

@app.route("/end_meeting", methods=["POST"])
def end_meeting():
    if request.method == "POST":
        audio_data = request.json.get("audio")
        if audio_data:
            print("Audio data received")
            try:
                audio_format = audio_data.split(";")[0].split("/")[-1]
                audio_data = audio_data.split(",")[1]
                audio_data = base64.b64decode(audio_data)

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as audio_file:
                    audio_file.write(audio_data)
                    audio_file_path = audio_file.name

                audio = AudioSegment.from_file(audio_file_path, format=audio_format)
                audio.export(audio_file_path[:-len(audio_format)-1] + ".wav", format="wav")

                r = sr.Recognizer()
                with sr.AudioFile(audio_file_path[:-5] + ".wav") as source:
                    audio = r.record(source)
                transcript = r.recognize_google(audio)
                print("Transcript:", transcript)
            except sr.RequestError as e:
                print("RequestError:", e)
                return jsonify(error="Could not request results; {0}".format(e)), 500

            except sr.UnknownValueError:
                print("UnknownValueError")
                return jsonify(error="Unknown error occurred"), 500

            except Exception as e:
                print("General Exception:", e)
                return jsonify(error="An error occurred while processing the audio."), 500

            try:
                MODEL = "gpt-3.5-turbo"
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
                        {"role": "user", "content": f"Summarize the following meeting transcript: {transcript}"}],
                    max_tokens=TOKENS,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                summary = response['choices'][0]['message']['content'].strip()
                print("Summary:", summary)
                return jsonify(summary=summary)

            except Exception as e:
                print(e)
                return jsonify(error="An error occurred while processing the audio."), 500

if __name__ == "__main__":
    app.run()
