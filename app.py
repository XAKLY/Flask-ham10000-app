import os
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import joblib
import uuid
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle SVM et l'encodeur de libellé
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Correspondance des labels codés aux noms complets
label_map = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

def predict_skin_lesion(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.asarray(img).flatten()
    prediction = svm_model.predict([img_array])
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    predicted_class = label_map.get(predicted_label, "Unknown")  # Utiliser le nom complet de la classe
    return predicted_class

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploads_dir = 'uploads'
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            unique_filename = str(uuid.uuid4()) + '.jpg'
            img_path = os.path.join(uploads_dir, unique_filename)
            uploaded_file.save(img_path)
            
            try:
                prediction = predict_skin_lesion(img_path)
            except ValueError as e:
                # Si une erreur se produit lors de la prédiction, renvoyer un message d'erreur convivial
                error_message = "Impossible to detect lesion on this image. please use another image."
                return error_message
                # return render_template('index.html', error=error_message)
            
            os.remove(img_path)
            return prediction
            # return render_template('result.html', prediction=prediction)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
