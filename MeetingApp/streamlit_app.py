import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)
from io import BytesIO
import requests

# URL of your the Flask app
FLASK_APP_URL = "https://meeting-summarizer.herokuapp.com/upload"

# Any code needed for the Streamlit app.

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.audio_buffer = BytesIO()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        frame.to_ndarray().tobytes(stream=self.audio_buffer)
        return None

def main():
    st.title('🎙️ Speech-to-Text Summarizer')
    st.write("Click 'Start Meeting' to begin recording. Click 'End Meeting' to stop recording and see the transcription and summary.")

    recorder = AudioRecorder()
    webrtc_ctx = webrtc_streamer(key="audio", mode=WebRtcMode.SENDONLY, client_settings=ClientSettings(rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}], "trickle": True}, media_stream_constraints={"video": False, "audio": True},), audio_processor_factory=AudioRecorder, audio_processor_factory_kwargs={})

    start_button = st.button("Start Meeting")
    stop_button = st.button("End Meeting")

    if stop_button:
    if webrtc_ctx.state.playing:
        webrtc_ctx.stop_all()
        recorder.audio_buffer.seek(0)
        audio_data = recorder.audio_buffer

        # Send the recorded audio to the Flask API
        response = requests.post(
            FLASK_APP_URL,
            files={"file": (f"audio.wav", audio_data, "audio/wav")},
        )

        if response.status_code == 200:
            result = response.json()
            transcription_text = result["transcription"]
            summary_text = result["summary"]

            st.write('**Transcription:**')
            st.write(transcription_text)

            st.write('**Summary:**')
            st.write(summary_text)

        else:
            st.error(f"Error: {response.status_code}, {response.text}")

    else:
        st.warning("Please start the meeting first before stopping it.")

if __name__ == "__main__":
    main()