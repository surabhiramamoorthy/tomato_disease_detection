from flask import Flask, render_template, request
from flask_babel import Babel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
from flask_babel import _  
from flask_babel import gettext as _

app = Flask(__name__)

# Initialize Babel for i18n
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations' 
babel = Babel(app)


# Set upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Locale selector function
def get_locale():
    lang = request.args.get('lang')
    print(f"Locale selected: {lang}")
    if lang in ['en', 'hi', 'kn']:
        return lang
    return app.config['BABEL_DEFAULT_LOCALE']

babel.locale_selector_func = get_locale

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
from flask_babel import _

disease_details = {
    "Bacterial_Spot": {
        "description": _("Bacterial spot is a common disease that causes dark, water-soaked spots on leaves and fruit."),
        "causes": _("Caused by Xanthomonas bacteria, often spread through infected seeds or water splash."),
        "symptoms": _("Small brown lesions on leaves, black spots on fruit, yellowing and early leaf drop."),
        "cure": _("Remove infected parts, use copper-based bactericides, avoid overhead watering.")
    },
    "Early_blight": {
        "description": _("Early blight causes concentric ring-like spots on older leaves."),
        "causes": _("Caused by Alternaria solani, a soil-borne fungus."),
        "symptoms": _("Brown circular spots with concentric rings, yellowing leaves, stem lesions."),
        "cure": _("Apply fungicides, crop rotation, and remove infected debris.")
    },
    "Late_blight": {
        "description": _("A serious disease that can destroy tomato crops rapidly under moist conditions."),
        "causes": _("Caused by Phytophthora infestans, a water mold pathogen."),
        "symptoms": _("Dark brown blotches on leaves with pale green halos, fruit rot."),
        "cure": _("Use fungicides, avoid wet foliage, destroy infected plants.")
    },
    "Leaf_Mold": {
        "description": _("Fungal disease causing pale green then yellow spots on upper leaf surface."),
        "causes": _("Caused by the fungus Passalora fulva, spreads in humid environments."),
        "symptoms": _("Velvety olive-green mold on underside of leaves, leaf curling and drop."),
        "cure": _("Use resistant varieties, improve ventilation, apply fungicides.")
    },
    "Septoria_leaf_spot": {
        "description": _("Common fungal disease that starts as small water-soaked circular spots."),
        "causes": _("Caused by the fungus Septoria lycopersici."),
        "symptoms": _("Tiny dark spots with gray centers, severe leaf loss if untreated."),
        "cure": _("Remove infected leaves, fungicide application, and avoid overhead watering.")
    },
    "Spider_mites Two-spotted_spider_mite": {
        "description": _("Tiny spider-like pests that suck plant juices and cause yellow stippling."),
        "causes": _("Infestation by two-spotted spider mites (Tetranychus urticae)."),
        "symptoms": _("Fine webbing, yellow or bronze leaf stippling, leaf curling."),
        "cure": _("Use miticides or insecticidal soap, maintain plant hydration.")
    },
    "Target_Spot": {
        "description": _("Fungal disease leading to dark concentric ring spots resembling targets."),
        "causes": _("Caused by Corynespora cassiicola fungus."),
        "symptoms": _("Spots with tan centers and dark margins, leaf drop, stem lesions."),
        "cure": _("Remove affected leaves, improve air circulation, apply fungicide.")
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": _("A viral disease that severely stunts tomato plants."),
        "causes": _("Transmitted by whiteflies."),
        "symptoms": _("Upward curling of leaves, yellowing, stunted growth."),
        "cure": _("Remove infected plants, control whiteflies, use resistant varieties.")
    },
    "Tomato_mosaic_virus": {
        "description": _("Virus that causes mosaic-like mottling on leaves and fruit distortion."),
        "causes": _("Caused by Tomato mosaic virus (ToMV)."),
        "symptoms": _("Mottled leaves, reduced fruit size, distorted fruit shape."),
        "cure": _("Remove infected plants, sanitize tools, avoid tobacco exposure.")
    },
    "Healthy": {
        "description": _("No disease detected! The leaf looks healthy."),
        "causes": _("N/A"),
        "symptoms": _("N/A"),
        "cure": _("Continue proper care and monitoring.")
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
    lang = get_locale()
    return render_template('index.html', lang=lang)


@app.route('/predict', methods=['POST'])
def predict():
    lang = request.args.get('lang', 'en')  # Get language from URL query, default to 'en'
    
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)[0]
    result = np.argmax(pred)
    confidence = float(pred[result]) * 100
    disease = disease_dict.get(result, "Unknown")

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
                           lang=lang,   # <-- pass lang here!
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