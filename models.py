from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from voice_assistant.voice_assistant import VoiceAssistant
from cultural_modern.water_conservation_analyzer import WaterConservationAnalyzer
from irrigation_plan.irrigation_recommender import irrigation_recommendation_engine, get_location_from_coords

from cultural_practices.app import predict_festival_and_practice

app = Flask(__name__)
CORS(app)

assistant_model = VoiceAssistant()
water_conservation_analyzer = WaterConservationAnalyzer()

@app.route('/')
def home():
    return "Hello, Flask is running on port 7000!"

@app.route('/assistant_response', methods=['POST'])
def predict():
    data = request.get_json()
    audio_path = data['audio_path']
    audio_response = assistant_model.forward(audio_path=audio_path)

    return jsonify({
        'audio_response': audio_response
    })

@app.route('/get_conversation_history', methods=['GET'])
def history():
    history = assistant_model.get_conversation_history()
    data = []
    for idx, message in enumerate(history):
        if idx%2 == 0:
            data.append({"User": message.content})
        else:
            data.append({"AI": message.content})
    return jsonify({
        'history': data
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

@app.route("/water_analysis", methods=['GET'])
def water_analyse():
    # data = request.json
    result = water_conservation_analyzer.analyze_practices()
    traditional_practice = result.traditional_practice
    traditional_efficiency = result.traditional_efficiency
    traditional_description = result.traditional_description
    modern_practice = result.modern_practice
    improved_efficiency = result.improved_efficiency
    modern_description = result.modern_description
    return jsonify({
        "traditional_practice": traditional_practice,
        "traditional_efficiency": traditional_efficiency,
        "traditional_description": traditional_description,
        "modern_practice": modern_practice,
        "improved_efficiency": improved_efficiency,
        "modern_description": modern_description
    })

@app.route('/predict_festival_practice', methods=['POST'])
def predict_festival_practice():
    data = request.get_json()
    transcript = data.get('transcript', '')
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    predicted_festival, predicted_practice = predict_festival_and_practice(transcript)
    return jsonify({
        'predicted_festival': predicted_festival,
        'predicted_practice': predicted_practice
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)


# {
#   "crop": "wheat",
#   "stage": "seedling",
#   "location": [40.7128, -74.0060]
# }