import sys
import os
from dotenv import load_dotenv
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'voice_assistant')))
from model_classes import LLMModel

model = LLMModel()

load_dotenv()

WEATHER_BASE_URL = os.getenv("WEATHER_BASE_URL")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather_data(location):
    lat, lon = location
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain": data.get("rain", {}).get("1h", 0),
            "wind_speed": data["wind"]["speed"]
        }
        print(f"[DEBUG] Weather Data for {location}: {weather_info}")  # Debug print
        return weather_info
    else:
        raise Exception(f"Failed to fetch weather data: {response.status_code}")



def calculate_irrigation(soil_type, temp, humidity, rain, wind_speed):
    base_water = {
        "sandy": 20,
        "sandy loam": 17,
        "loamy": 14,
        "silt loam": 12,
        "clay loam": 10,
        "clay": 9,
        "peaty": 16,
        "volcanic loam": 14,
        "silty clay loam": 12,
        "sandy clay loam": 13,
        "silty": 10,
        "clayey silt": 9,
        "alluvial loam": 12,
        "chernozem": 14,
        "glacial till and rocky": 8,
        "rocky and sandy": 6
    }.get(soil_type.lower(), 12)  # Default to 12L/m² if unknown

    # **Evapotranspiration (ET) Factor Adjustments**
    temp_factor = 1 + (temp - 20) * 0.08  # 8% change per °C deviation from 20°C
    humidity_factor = 1 - (humidity - 50) * 0.03  # 3% reduction per 10% humidity increase
    wind_factor = 1 + (wind_speed * 0.04)  # 4% increase per m/s wind speed

    # **Final irrigation requirement**
    adjusted_water = base_water * temp_factor * humidity_factor * wind_factor - rain
    final_water = max(5, round(adjusted_water, 1))  # Ensure a minimum of 5L/m²

    # Debugging info
    print(f"[DEBUG] {soil_type} Soil - Base Water: {base_water}L | Adjusted: {final_water}L")
    print(f"[DEBUG] Factors - Temp: {temp_factor:.2f}, Humidity: {humidity_factor:.2f}, Wind: {wind_factor:.2f}, Rain Reduction: {rain}mm")

    return final_water


def generate_irrigation_plan(crop_type, growth_stage, soil_type, weather, exact_location):
    irrigation_liters = calculate_irrigation(
        soil_type, weather["temp"], weather["humidity"], weather["rain"], weather["wind_speed"]
    )

    irrigation_plan = (
        f"Exact Location to be mentioned: {exact_location}"
        f"For {crop_type} in the {growth_stage} stage growing in {soil_type} soil, "
        f"temperature: {weather['temp']}K, humidity: {weather['humidity']}%, "
        f"rain: {weather['rain']}mm, wind: {weather['wind_speed']}m/s → "
        f"Water {irrigation_liters} liters per square meter daily at 6 AM."
    )

    text = f"Exact Location to be mentioned: {exact_location}. Generate an irrigation plan with minimal water wastage for {crop_type}. Method to be mentioned: Give the most suitable, most efficient technology for irrigation. For {crop_type} in the {growth_stage} stage growing in {soil_type} soil temperature: {weather['temp']}K, humidity: {weather['humidity']} rain: {weather['rain']}mm, wind: {weather['wind_speed']}m/s"
    
    result = model.get_response(text, long_context=True)

    return result

def get_location_from_coords(location):
    lat, lon = location
    limit = 4
    API_URL = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit={limit}&appid={WEATHER_API_KEY}"
    response = requests.get(API_URL)
    exact_location = response.json()[0]['name']
    return exact_location
    

def irrigation_recommendation_engine(crop_type, growth_stage, location, exact_location):
    weather = get_weather_data(location)
    # soil_type = get_soil_data(location)
    soil_type = "loamy"
    plan = generate_irrigation_plan(crop_type, growth_stage, soil_type, weather, exact_location)
    return plan

if __name__ == "__main__":
    crop = "potatoes"
    stage = "vegetative"
    location_coords = [22.2604, 84.8536]
    exact_location = get_location_from_coords(location_coords)
    
    try:
        result = irrigation_recommendation_engine(crop, stage, location_coords, exact_location)
        print("\n🌿 **Irrigation Plan:**")
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
