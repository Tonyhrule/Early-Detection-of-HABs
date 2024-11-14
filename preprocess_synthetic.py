import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()

# Configuration Parameters
DATA_PATH = 'Dataset.xlsx'
OUTPUT_DIR = 'output'
SYNTHETIC_DATA_BATCHES = 3
SYNTHETIC_DATA_ROWS = 100
TEMP_RANGE = (15, 20)
SALINITY_RANGE = (34, 35)
UVB_RANGE = (30, 80)
CHLOROPHYLL_RANGE = (5, 15)
TEST_SIZE = 0.2
RANDOM_STATE = 42

def chatgpt_response(messages, model="gpt-4", retries=3, timeout=300):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(model=model,
            messages=messages,
            timeout=timeout)
            return response.choices[0].message.content
        except openai.Timeout:
            print(f"Request timed out. Attempt {attempt + 1} of {retries}. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return None

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
    return pd.read_excel(data_path)

def clean_data(data):
    return data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])

def generate_synthetic_data():
    prompt_content = (
        "Generate synthetic data rows with 'Temperature', 'Salinity', 'UVB', and 'ChlorophyllaFlor' columns. "
        "Temperature should range from 15 to 20 degrees Celsius, Salinity from 34 to 35 PSU, "
        "UVB between 30 and 80 mW/m², and ChlorophyllaFlor between 5 and 15 µg/L. "
        f"Generate {SYNTHETIC_DATA_ROWS} rows."
    )

    messages = [
        {"role": "system", "content": "You are a data generator for environmental data related to harmful algal blooms."},
        {"role": "user", "content": prompt_content}
    ]

    synthetic_data_list = []
    for i in range(SYNTHETIC_DATA_BATCHES):
        response_content = chatgpt_response(messages=messages, model="gpt-4")
        if response_content:
            try:
                string_data = StringIO(response_content)
                generated_data = pd.read_csv(string_data, sep=',')
                synthetic_data_list.append(generated_data)
            except Exception as e:
                print(f"Error parsing synthetic data: {e}")

    if synthetic_data_list:
        return pd.concat(synthetic_data_list, axis=0, ignore_index=True)
    else:
        raise ValueError("No synthetic data generated. Check generation process.")

def preprocess_and_combine(data, synthetic_data):
    combined_data = pd.concat([data, synthetic_data], ignore_index=True)
    X = combined_data[['Temperature', 'Salinity', 'UVB']]
    y = combined_data['ChlorophyllaFlor']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data_with_synthetic.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_with_synthetic.pkl'))

def main():
    data = load_data(DATA_PATH)
    data = clean_data(data)
    synthetic_data = generate_synthetic_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_combine(data, synthetic_data)
    save_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)
    print("Preprocessing with synthetic data completed successfully.")

if __name__ == "__main__":
    main()
