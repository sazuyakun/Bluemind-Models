from flask import Flask, request, jsonify
from voice_assistant.voice_assistant import VoiceAssistant

app = Flask(__name__)

model = VoiceAssistant()

@app.route('/')
def home():
    return "Hello, Flask is running on port 7000!"

@app.route('/assistant_response', methods=['POST'])
def predict():
    data = request.get_json()
    audio_path = data['audio_path']
    audio_response = model.forward(audio_path=audio_path)

    return jsonify({
        'audio_response': audio_response
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)