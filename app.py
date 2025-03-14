import re
import os
import cv2
import requests
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from keras.models import load_model
from services.module_one import flask_api_route
from services.module_two import extract_text_from_image
from flask import Flask, request, jsonify, render_template


# Load the trained CNN model
model = load_model("Dataset Generator/product_classification_cnn.h5")

# Define class names
class_names = ['6 RIBBONS RUSTIC CHARM', 'ALARM CLOCK BAKELIKE RED', 'CHOCOLATE HOT WATER BOTTLE', 
               'JUMBO STORAGE BAG SUKI', 'LUNCH BAG PINK POLKADOT', 'LUNCH BAG WOODLAND', 
               'REGENCY CAKESTAND 3 TIER', 'RETROSPOT TEA SET CERAMIC 11 PC', 
               'REX CASHCARRY JUMBO SHOPPER', 'SPOTTY BUNTING']

# Initialize Flask app
app = Flask(__name__)
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def generate_response(query):
    """
    Sends a query to the DeepSeek API and returns a natural language response.
    """
    system_prompt = (
        "You are a helpful assistant that provides direct and concise answers without asking for clarification. "
        "If the user mentions a product, describe it Do not ask follow-up questions."
        " for example if the user types in headphones the reply should be Experience exceptional sound with our High-Quality Headphones, "
        "featuring advanced noise cancellation and a comfortable over-ear design, perfect for any audio enthusiast."
        "Return only text, and no special styling or formatting is required."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define the payload for the API request
    payload = {
        "model": "deepseek-chat",  # Replace with the appropriate model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 150  # Adjust as needed
    }

    # Send the request to the DeepSeek API
    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"
    
# Safeguard: Validate the input query
def validate_query(query):
    """
    Validate the natural language query to ensure it's safe and meaningful.
    """
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string."
    
    # Check for malicious input (e.g., SQL injection, special characters)
    if re.search(r'[;\-\-\\/*]', query):
        return False, "Invalid characters in query."
    
    return True, "Query is valid."

# Home page with a form for query input
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Endpoint 1: Product Recommendation Service
@app.route('/recommend', methods=['POST'])
def recommend_products():
    """
    Handle natural language queries to recommend products.
    Input: Form data with a "query" field.
    Output: Renders a template with product matches and a natural language response.
    """
    # Get the query from the form
    query = request.form.get('query')
    
    # Validate the query
    is_valid, message = validate_query(query)
    if not is_valid:
        return render_template('index.html', error=message)
    
    try:
        
        # Step 4: Generate a natural language response
        matches = flask_api_route(query)
        response = generate_response(query)
    
        # Step 5: Render the template with results
        return render_template('index.html', response=response, matches=matches)
    
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route("/ocr", methods=["GET", "POST"])
def index():
    """
    Renders the index page and handles image uploads.
    """
    extracted_text = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No image file provided", 400
        
        # Save the uploaded image
        image_file = request.files["image"]
        image_path = "uploaded_image.png"
        image_file.save(image_path)
        
        # Extract text using OCR
        extracted_text = extract_text_from_image(image_path)
    
    # Render the template with the extracted text
        matches = flask_api_route(extracted_text)
        response = generate_response(extracted_text)

    # Step 5: Render the template with results
        return render_template('ocr.html', response=response, matches=matches, extracted_text=extracted_text)
    return render_template('ocr.html')

# Define the prediction endpoint
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Endpoint to handle image upload and prediction.
    """
    predicted_class = None

    if request.method == "POST":
        # Check if an image file was uploaded
        if "image" not in request.files:
            return "No image file provided", 400
        
        # Read the image file

        image_file = request.files["image"]
        image_path = "uploaded_image.png"
        image_file.save(image_path)
        img = cv2.imread(image_path)
        resize = tf.image.resize(img, (128,128))

        # Make a prediction
        predictions = model.predict(np.expand_dims(resize/255, 0))
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        print(f'this is the predicted class {predicted_class}')

        matches = flask_api_route(predicted_class)
        response = generate_response(predicted_class)
        return render_template('cnn.html', response=response, matches=matches, extracted_text=predicted_class)

    # Render the template with the result
    return render_template("cnn.html", predicted_class=predicted_class)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)