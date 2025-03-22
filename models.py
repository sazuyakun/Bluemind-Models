from flask import Flask, request, jsonify
from voice_assistant.voice_assistant import VoiceAssistant
from irrigation_plan.irrigation_recommender import irrigation_recommendation_engine, get_location_from_coords

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


@app.route('/irrigation_plan', methods=['POST'])
def irrigation_recommend():
    data = request.get_json()
    crop = data['crop']
    stage = data['stage']
    location = data['location']
    exact_location = get_location_from_coords(location)
    
    result = irrigation_recommendation_engine(crop, stage, location, exact_location)
    final_result = f'''
    \n🌿 **Irrigation Plan:**\n
    {result}
    '''
    return jsonify({
        'irrigation_plan': final_result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)