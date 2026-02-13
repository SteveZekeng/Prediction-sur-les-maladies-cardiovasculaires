# importation des bibliotheques
from flask import Flask, render_template, request
import pandas as pd
from utils import load_models, preprocess_input, evaluate_models

app = Flask(__name__)

# Load trained models and their KPIs
models = load_models()

@app.route('/')
def index():
     return render_template('index.html') # retoune notre page web 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input data from form
        input_data = {key: float(request.form[key]) for key in request.form}

        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        X_processed = preprocess_input(input_df)

        # Make predictions and evaluate
        predictions, kpis = evaluate_models(models, X_processed)

        return render_template(
            'results.html',
            input_data=input_data,
            predictions=predictions,
            kpis=kpis
        )

if __name__ == '__main__':
    app.run(debug=True)
