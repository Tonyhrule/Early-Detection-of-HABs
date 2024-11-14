import os
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from dotenv import load_dotenv
from autogen.agentchat import AssistantAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

models_dir = 'models'
output_dir = 'output'

# Select between base or synthetic model
synthetic = input("Use synthetic-augmented model? (yes/no): ").strip().lower() in ["yes", "y"]
scaler_path = os.path.join(output_dir, 'scaler_with_synthetic.pkl' if synthetic else 'scaler.pkl')
model_path = os.path.join(models_dir, 'stacker_synthetic_model.pkl' if synthetic else 'stacker_model.pkl')
processed_data_path = os.path.join(output_dir, 'processed_data_with_synthetic.pkl' if synthetic else 'processed_data.pkl')

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(processed_data_path)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly.fit(X_train_scaled)

def predict_chlorophyll(temperature, salinity, uvb):
    features = pd.DataFrame([[float(temperature), float(salinity), float(uvb)]], columns=['Temperature', 'Salinity', 'UVB'])
    features_scaled = scaler.transform(features.values)  # Convert to array to avoid warning
    features_poly = poly.transform(features_scaled)
    return model.predict(features_poly)[0]

def categorize_chlorophyll_a(value):
    if value < 2:
        return "Low Risk (0-5% Probability)"
    elif 2 <= value < 7:
        return "Moderate Risk (5-25% Probability)"
    elif 7 <= value < 12:
        return "High Risk (25-50% Probability)"
    else:
        return "Very High Risk (50-100% Probability)"

def main():
    temperature = float(input("Enter water temperature: "))
    salinity = float(input("Enter salinity level: "))
    uvb = float(input("Enter UVB level: "))

    chlorophyll_a_value = predict_chlorophyll(temperature, salinity, uvb)
    risk_category = categorize_chlorophyll_a(chlorophyll_a_value)

    print(f"Predicted Chlorophyll a Corrected: {chlorophyll_a_value:.2f} µg/L - {risk_category}")

    if input("Would you like to consult the assistant for advice? (yes/no): ").strip().lower() in ["yes", "y"]:
        assistant_response(chlorophyll_a_value, risk_category)

def assistant_response(chlorophyll_a_value, risk_category):
    config_list = [{"model": "gpt-4", "api_key": openai_api_key}]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides insights on Harmful Algal Bloom risk and prevention."},
        {"role": "user", "content": f"## Chlorophyll A corrected value: {chlorophyll_a_value:.2f} µg/L\n## Risk Category: {risk_category}\n\nWhat should be done to prevent Harmful Algal Blooms?"}
    ]

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are an assistant providing information on Harmful Algal Bloom risk and mitigation.",
        llm_config={"timeout": 600, "seed": 42, "config_list": config_list}
    )

    response = assistant.generate_reply(messages)
    print("\nAssistant Response:")
    print(response)

    if input("Would you like a more tailored response? (yes/no): ").strip().lower() in ["yes", "y"]:
        weather = input("Current weather: ")
        season = input("Current season: ")
        water_flow = input("Water flow (e.g., still, slow, fast): ")

        tailored_message = {
            "role": "user",
            "content": (f"Given these conditions:\nWeather: {weather}\nSeason: {season}\nWater Flow: {water_flow}\n"
                        f"The Chlorophyll A corrected value is {chlorophyll_a_value:.2f} µg/L ({risk_category}). "
                        "What actions are recommended to prevent Harmful Algal Blooms?")
        }
        tailored_response = assistant.generate_reply(messages + [tailored_message])
        print("\nTailored Assistant Response:")
        print(tailored_response)

if __name__ == "__main__":
    main()
