import os
import time
import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables
load_dotenv()

# Configuration Parameters
DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
SYNTHETIC_DATA_BATCHES = 4
SYNTHETIC_DATA_ROWS = 50
TEMP_RANGE = (15, 20)
SALINITY_RANGE = (34, 35)
UVB_RANGE = (30, 80)
CHLOROPHYLL_RANGE = (5, 15)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Expected columns for the dataset
EXPECTED_COLUMNS = ["Temperature", "Salinity", "UVB", "ChlorophyllaFlor"]

def load_data(data_path):
    """Loads the dataset from an Excel file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
    print("Loading dataset...")
    return pd.read_excel(data_path)

def validate_structure(df):
    """Validates that the DataFrame matches the expected structure."""
    if not all(col in df.columns for col in EXPECTED_COLUMNS):
        raise ValueError(f"DataFrame columns do not match expected structure: {EXPECTED_COLUMNS}")
    print("Dataset structure validated.")

def generate_synthetic_data(batch_size, rows_per_batch):
    """Generates synthetic data using GPT-4 with structured output."""
    synthetic_data_list = []

    for _ in range(batch_size):
        print("Generating synthetic data batch...")
        try:
            response = client.chat.completions.create(model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a data generator for environmental science."},
                {"role": "user", "content": f"Generate {rows_per_batch} rows of structured environmental data with the following ranges:\n"
                                             f"- Temperature: {TEMP_RANGE[0]} to {TEMP_RANGE[1]} (°C)\n"
                                             f"- Salinity: {SALINITY_RANGE[0]} to {SALINITY_RANGE[1]} (PSU)\n"
                                             f"- UVB: {UVB_RANGE[0]} to {UVB_RANGE[1]} (mW/m²)\n"
                                             f"- ChlorophyllaFlor: {CHLOROPHYLL_RANGE[0]} to {CHLOROPHYLL_RANGE[1]} (µg/L)\n"
                                             "Return the data as an array of JSON objects with keys: Temperature, Salinity, UVB, ChlorophyllaFlor."}
            ],
            functions=[
                {
                    "name": "generate_data",
                    "description": "Generate structured environmental data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Temperature": {"type": "number", "minimum": TEMP_RANGE[0], "maximum": TEMP_RANGE[1]},
                                        "Salinity": {"type": "number", "minimum": SALINITY_RANGE[0], "maximum": SALINITY_RANGE[1]},
                                        "UVB": {"type": "number", "minimum": UVB_RANGE[0], "maximum": UVB_RANGE[1]},
                                        "ChlorophyllaFlor": {"type": "number", "minimum": CHLOROPHYLL_RANGE[0], "maximum": CHLOROPHYLL_RANGE[1]}
                                    },
                                    "required": ["Temperature", "Salinity", "UVB", "ChlorophyllaFlor"]
                                }
                            }
                        },
                        "required": ["data"]
                    }
                }
            ])

            # Extract the function call arguments and parse as JSON
            function_call_arguments = response.choices[0].message.function_call.arguments
            structured_data = json.loads(function_call_arguments)["data"]
            batch_df = pd.DataFrame(structured_data)

            # Validate and append batch
            validate_structure(batch_df)
            synthetic_data_list.append(batch_df)
            print("Synthetic data batch generated successfully.")
        except Exception as e:
            print(f"Error generating synthetic data batch: {e}")
            continue

    if synthetic_data_list:
        return pd.concat(synthetic_data_list, ignore_index=True)
    else:
        raise ValueError("No synthetic data generated. Check the generation process.")

def preprocess_and_combine(data, synthetic_data):
    """Preprocesses and combines real and synthetic data."""
    combined_data = pd.concat([data, synthetic_data], ignore_index=True)

    X = combined_data[['Temperature', 'Salinity', 'UVB']]
    y = combined_data['ChlorophyllaFlor']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    """Saves processed data and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data_with_synthetic.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_with_synthetic.pkl'))

def main():
    """Main function to execute the synthetic preprocessing pipeline."""
    data = load_data(DATA_PATH)
    validate_structure(data)  # Validate real dataset structure
    synthetic_data = generate_synthetic_data(SYNTHETIC_DATA_BATCHES, SYNTHETIC_DATA_ROWS)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_combine(data, synthetic_data)
    save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)
    print("Preprocessing with synthetic data completed successfully.")

if __name__ == "__main__":
    main()
