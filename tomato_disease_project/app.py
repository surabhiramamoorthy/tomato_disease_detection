from flask import Flask, render_template, request
from flask_babel import Babel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
from flask_babel import _  

app = Flask(__name__)

# Initialize Babel for i18n
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(app)

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Locale selector function
def get_locale():
    return request.args.get('lang', 'en')

babel.init_app(app, locale_selector=get_locale)

# Load model
model = load_model("model_new_dataset.h5")

# Class dictionary for disease names
disease_dict = {
    0: "Bacterial_Spot",
    1: "Early_blight",
    2: "Late_blight",
    3: "Leaf_Mold",
    4: "Septoria_leaf_spot",
    5: "Spider_mites Two-spotted_spider_mite",
    6: "Target_Spot",
    7: "Tomato_Yellow_Leaf_Curl_Virus",
    8: "Tomato_mosaic_virus",
    9: "Healthy"
}

# Healthy ranges for sensor inputs
healthy_ranges = {
    "moisture": (30, 70),   # percentage
    "ph": (6.0, 6.8),       # ideal tomato pH
    "salinity": (1.0, 2.5)  # dS/m
}

# Disease information dictionary
disease_details = {
    "Bacterial_Spot": {
        "description": "Bacterial spot is a common disease that causes dark, water-soaked spots on leaves and fruit.",
        "causes": "Caused by Xanthomonas bacteria, often spread through infected seeds or water splash.",
        "symptoms": "Small brown lesions on leaves, black spots on fruit, yellowing and early leaf drop.",
        "cure": "Remove infected parts, use copper-based bactericides, avoid overhead watering."
    },
    "Early_blight": {
        "description": "Early blight causes concentric ring-like spots on older leaves.",
        "causes": "Caused by Alternaria solani, a soil-borne fungus.",
        "symptoms": "Brown circular spots with concentric rings, yellowing leaves, stem lesions.",
        "cure": "Apply fungicides, crop rotation, and remove infected debris."
    },
    "Late_blight": {
        "description": "A serious disease that can destroy tomato crops rapidly under moist conditions.",
        "causes": "Caused by Phytophthora infestans, a water mold pathogen.",
        "symptoms": "Dark brown blotches on leaves with pale green halos, fruit rot.",
        "cure": "Use fungicides, avoid wet foliage, destroy infected plants."
    },
    "Leaf_Mold": {
        "description": "Fungal disease causing pale green then yellow spots on upper leaf surface.",
        "causes": "Caused by the fungus Passalora fulva, spreads in humid environments.",
        "symptoms": "Velvety olive-green mold on underside of leaves, leaf curling and drop.",
        "cure": "Use resistant varieties, improve ventilation, apply fungicides."
    },
    "Septoria_leaf_spot": {
        "description": "Common fungal disease that starts as small water-soaked circular spots.",
        "causes": "Caused by the fungus Septoria lycopersici.",
        "symptoms": "Tiny dark spots with gray centers, severe leaf loss if untreated.",
        "cure": "Remove infected leaves, fungicide application, and avoid overhead watering."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny spider-like pests that suck plant juices and cause yellow stippling.",
        "causes": "Infestation by two-spotted spider mites (Tetranychus urticae).",
        "symptoms": "Fine webbing, yellow or bronze leaf stippling, leaf curling.",
        "cure": "Use miticides or insecticidal soap, maintain plant hydration."
    },
    "Target_Spot": {
        "description": "Fungal disease leading to dark concentric ring spots resembling targets.",
        "causes": "Caused by Corynespora cassiicola fungus.",
        "symptoms": "Spots with tan centers and dark margins, leaf drop, stem lesions.",
        "cure": "Remove affected leaves, improve air circulation, apply fungicide."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A viral disease that severely stunts tomato plants.",
        "causes": "Transmitted by whiteflies.",
        "symptoms": "Upward curling of leaves, yellowing, stunted growth.",
        "cure": "Remove infected plants, control whiteflies, use resistant varieties."
    },
    "Tomato_mosaic_virus": {
        "description": "Virus that causes mosaic-like mottling on leaves and fruit distortion.",
        "causes": "Caused by Tomato mosaic virus (ToMV).",
        "symptoms": "Mottled leaves, reduced fruit size, distorted fruit shape.",
        "cure": "Remove infected plants, sanitize tools, avoid tobacco exposure."
    },
    "Healthy": {
        "description": "No disease detected! The leaf looks healthy.",
        "causes": "N/A",
        "symptoms": "N/A",
        "cure": "Continue proper care and monitoring."
    }
}

def evaluate_sensor(value, min_val, max_val):
    try:
        value = float(value)  # Ensure value is numeric
    except (ValueError, TypeError):
        return _("Invalid input")

    if value < min_val:
        return _("Low")
    elif value > max_val:
        return _("High")
    else:
        return _("Normal")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))  # update target size if needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # normalize

    # Predict
    pred = model.predict(x)[0]
    result = np.argmax(pred)
    confidence = float(pred[result]) * 100

    disease = disease_dict.get(result, "Unknown")

    # Sensor inputs
    moisture = request.form.get('moisture')
    ph = request.form.get('ph')
    salinity = request.form.get('salinity')

    if not moisture or not ph or not salinity:
        return "Missing sensor inputs."

    try:
        moisture = float(moisture)
        ph = float(ph)
        salinity = float(salinity)
    except ValueError:
        return "Invalid sensor inputs."

    moisture_status = evaluate_sensor(moisture, 10, 60)
    ph_status = evaluate_sensor(ph, 5.5, 7.5)
    salinity_status = evaluate_sensor(salinity, 1.0, 2.5)

    return render_template('result.html',
                           disease=disease,
                           confidence=round(confidence, 2),
                           disease_info=disease_details.get(disease),
                           img_path=filepath,
                           moisture=moisture,
                           ph=ph,
                           salinity=salinity,
                           moisture_status=moisture_status,
                           ph_status=ph_status,
                           salinity_status=salinity_status)

if __name__ == '__main__':
    app.run(debug=True)