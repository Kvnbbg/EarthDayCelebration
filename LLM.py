import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask
import requests
import smtplib
from email.mime.text import MIMEText
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
from dotenv import load_dotenv

# Load environment variables for sensitive data
load_dotenv()

# Simulated "system prompt" for storing learned strategies (inspired by Karpathy's idea)
SYSTEM_PROMPT_FILE = "system_prompt.json"

def load_system_prompt():
    """Load the system prompt from a JSON file or initialize an empty one."""
    if os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error: System prompt file is corrupted. Initializing a new one.")
            return {"strategies": []}
    return {"strategies": []}

def save_system_prompt(prompt):
    """Save the system prompt to a JSON file."""
    try:
        with open(SYSTEM_PROMPT_FILE, 'w') as f:
            json.dump(prompt, f, indent=4)
    except Exception as e:
        print(f"Error saving system prompt: {e}")

def check_strategy(prompt, strategy_key):
    """Check if a strategy exists in the system prompt."""
    for strategy in prompt["strategies"]:
        if strategy["key"] == strategy_key:
            return strategy["value"]
    return None

def add_strategy(prompt, strategy_key, strategy_value):
    """Add a new strategy to the system prompt."""
    prompt["strategies"].append({"key": strategy_key, "value": strategy_value})
    save_system_prompt(prompt)

# 1. Data Manipulation with Pandas
def load_data(file_path='example_data.csv'):
    """Load data from a CSV file with error handling."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# 2. Machine Learning with Scikit-Learn and TensorFlow
def train_model(data):
    """Train a neural network model with learned strategies."""
    if data is None or 'target' not in data.columns:
        print("Error: Data not loaded or 'target' column not found.")
        return None, None

    # Load system prompt for learned strategies
    system_prompt = load_system_prompt()

    # Prepare data
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for a learned strategy on feature scaling
    scaling_strategy = check_strategy(system_prompt, "feature_scaling")
    if scaling_strategy:
        print(f"Applying learned strategy: {scaling_strategy}")
        if scaling_strategy == "apply StandardScaler to features":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    else:
        # Experiment with and without scaling to learn the best strategy
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train without scaling
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history_no_scaling = model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
        no_scaling_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        # Train with scaling
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history_with_scaling = model.fit(X_train_scaled, y_train, epochs=5, validation_split=0.2, verbose=0)
        with_scaling_acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]

        # Learn and store the better strategy
        if with_scaling_acc > no_scaling_acc:
            add_strategy(system_prompt, "feature_scaling", "apply StandardScaler to features")
            print("Learned strategy: Apply feature scaling with StandardScaler.")
            X_train, X_test = X_train_scaled, X_test_scaled
        else:
            add_strategy(system_prompt, "feature_scaling", "no scaling needed")
            print("Learned strategy: No feature scaling needed.")

    # Final training with the best strategy
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    return model, history

# 3. Data Visualization with Matplotlib and Seaborn
def visualize_data(data, feature_column='feature'):
    """Visualize the distribution of a specified feature."""
    if data is None or feature_column not in data.columns:
        print(f"Error: Data not loaded or '{feature_column}' column not found.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature_column], bins=30)
    plt.title(f'{feature_column.capitalize()} Distribution')
    plt.xlabel(f'{feature_column.capitalize()} Value')
    plt.ylabel('Count')
    plt.show()

# 4. Web Development with Flask
app = Flask(__name__)

@app.route('/')
def home():
    """Basic Flask endpoint to return a greeting."""
    return "Hello, World! from the Data Science App!"

def run_flask():
    """Run the Flask app (call this separately if needed)."""
    app.run(debug=False, host='0.0.0.0', port=5000)

# 5. API Request with Requests
def fetch_api_data(url='https://api.example.com/data'):
    """Fetch data from an API with error handling."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        print("API data retrieved successfully.")
        print(data)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data: {e}")
        return None
    except ValueError:
        print("Error: API response is not valid JSON.")
        return None

# 6. Sending Email with SMTPLib
def send_email(subject, body, to_email):
    """Send an email using SMTP with credentials from environment variables."""
    from_email = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.example.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))

    if not all([from_email, password]):
        print("Error: Email credentials not found in environment variables.")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# 7. Google Sheets with Gspread
def update_google_sheet():
    """Update a Google Sheet with a sample value."""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open("Example Sheet").sheet1
        sheet.update_cell(1, 1, "Hello, World! from Data Science App")
        print("Google Sheet updated successfully.")
    except FileNotFoundError:
        print("Error: 'credentials.json' not found.")
    except gspread.exceptions.SpreadsheetNotFound:
        print("Error: Google Sheet 'Example Sheet' not found.")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

# 8. Current Date and Time
def print_current_time():
    """Print the current date and time in a readable format."""
    current_time = datetime.datetime.now()
    print("Current date and time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

# Main execution
def main():
    """Main function to orchestrate the script's execution."""
    # Step 1: Load data
    data = load_data()

    # Step 2: Train model with learned strategies
    model, history = train_model(data)

    # Step 3: Visualize data
    visualize_data(data)

    # Step 4: Fetch API data
    fetch_api_data()

    # Step 5: Send an email
    send_email("Data Science Update", "Model training completed successfully!", "recipient@example.com")

    # Step 6: Update Google Sheet
    update_google_sheet()

    # Step 7: Print current time
    print_current_time()

    # Step 8: Flask app setup (run separately if needed)
    print("Flask app is set up but not running. Call run_flask() to start it.")

if __name__ == "__main__":
    main()