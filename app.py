import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from ultralytics import YOLO
import xgboost as xgb
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Load dataset and models (make sure paths are correct)
file_path = 'D:\\WT material\\wt cp ui\\ui\\venv\\my recipe.csv'
recipe_df = pd.read_csv(file_path)

recipe_df['ingredients_list'] = recipe_df['ingredients_list'].fillna('')

imputer = SimpleImputer(strategy='mean')

# Extract numerical features and ingredients from your dataset
numerical_features = recipe_df[['calories', 'fat', 'carbohydrates', 'protein']]
ingredients = recipe_df['ingredients_list']
target = recipe_df['avrg_rate']  # Assuming there's a column for user ratings or preferences


numerical_features_imputed = imputer.fit_transform(numerical_features)

scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_features_imputed)

vectorizer = TfidfVectorizer()
transformed_ingredients = vectorizer.fit_transform(ingredients).toarray()

combined_features = np.hstack([scaled_numerical, transformed_ingredients])

# Train KNN
knn = NearestNeighbors(n_neighbors=5)
knn.fit(combined_features)

X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror')
xgboost_model.fit(X_train, y_train)

# YOLOv10 Model for Ingredient Detection
model = YOLO('D:\\WT material\\wt cp ui\\ui\\venv\\ingred.pt')  # Replace with actual path to YOLO model

def predict_ingredients(image_path):
    results = model(image_path)
    predicted_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls)  # Extract class ID
        class_name = model.names[class_id]  # Retrieve class name based on ID
        predicted_classes.append(class_name)
    
    predicted_classes = list(set(predicted_classes))  # Remove duplicates
    return predicted_classes

def recommend_recipes(input_features):
    numerical_features = input_features[:4]
    predicted_ingredients = input_features[4]

    input_features_imputed = imputer.transform([numerical_features])
    input_features_scaled = scaler.transform(input_features_imputed)

    input_ingredients_transformed = vectorizer.transform([predicted_ingredients])

    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])

    distances, indices = knn.kneighbors(input_combined)

    knn_recommendations = recipe_df.iloc[indices[0]]

    knn_features = np.hstack([
        scaler.transform(imputer.transform(knn_recommendations[['calories', 'fat', 'carbohydrates', 'protein']])),
        vectorizer.transform(knn_recommendations['ingredients_list']).toarray()
    ])

    xgboost_scores = xgboost_model.predict(knn_features)
    knn_recommendations['xgboost_score'] = xgboost_scores

    re_ranked_recommendations = knn_recommendations.sort_values(by='xgboost_score', ascending=False)

    return re_ranked_recommendations[['recipe_name', 'ingredients_list', 'image_url', 'xgboost_score']]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print("Image upload request received")  # Confirm request received
    
    if 'image' not in request.files:
        print("No image uploaded")  # Log if image is missing
        return jsonify({'error': 'No image uploaded'}), 400

    # Get the uploaded image and save it
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)
    print(f"Image saved at {image_path}")  # Log image path

    # Get the form data
    try:
        calories = float(request.form['calories'])
        fats = float(request.form['fats'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
    except ValueError as e:
        print(f"Form data conversion error: {e}")
        return jsonify({'error': 'Invalid form data'}), 400

    print(f"Received nutrition data: Calories={calories}, Fats={fats}, Carbs={carbohydrates}, Protein={protein}")

    # Predict ingredients using YOLO
    predicted_ingredients = predict_ingredients(image_path)
    print(f"Predicted ingredients: {predicted_ingredients}")

    predicted_ingredients_str = ', '.join(predicted_ingredients)  # Join ingredients as a string

    # Combine input features
    input_features = [calories, fats, carbohydrates, protein, predicted_ingredients_str]

    # Get recipe recommendations
    recommendations = recommend_recipes(input_features)
    print(f"Number of recommendations: {len(recommendations)}")

    # Convert recommendations to list of dicts for JSON response
    recommendations_list = recommendations.to_dict(orient='records')
    
    return jsonify(recommendations_list)

# Test route for hardcoded recommendations
@app.route('/test', methods=['GET'])
def test_recommendations():
    input_features = [200, 10, 50, 30, 'tomato, onion, garlic']  # Sample input
    recommendations = recommend_recipes(input_features)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    # Create the uploads directory if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
