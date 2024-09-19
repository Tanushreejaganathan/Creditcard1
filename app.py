from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model, encoder, and scaler
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input data from the form
            amount = request.form.get('amount')
            location = request.form.get('location')
            device_type = request.form.get('device_type')
            age = request.form.get('age')
            income = request.form.get('income')
            debt = request.form.get('debt')
            credit_score = request.form.get('credit_score')

            # Debugging output
            print(f"Amount: {amount}")
            print(f"Location: {location}")
            print(f"Device Type: {device_type}")
            print(f"Age: {age}")
            print(f"Income: {income}")
            print(f"Debt: {debt}")
            print(f"Credit Score: {credit_score}")

            # Check for missing values and convert to appropriate types
            if None in (amount, location, device_type, age, income, debt, credit_score):
                return "Error: Missing data in the form. Please fill out all fields.", 400

            amount = float(amount)
            age = float(age)
            income = float(income)
            debt = float(debt)
            credit_score = float(credit_score)

            # Prepare the input data
            input_data = pd.DataFrame({
                'amount': [amount],
                'location': [location],
                'device_type': [device_type],
                'age': [age],
                'income': [income],
                'debt': [debt],
                'credit_score': [credit_score]
            })

            # Encode categorical features
            input_data_encoded = encoder.transform(input_data[['location', 'device_type']])
            input_data_encoded = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out())

            # Standardize numerical features
            numerical_features = ['amount', 'age', 'income', 'debt', 'credit_score']
            input_data_scaled = scaler.transform(input_data[numerical_features])
            input_data_scaled = pd.DataFrame(input_data_scaled, columns=numerical_features)

            # Combine encoded and scaled features
            input_data_prepared = pd.concat([input_data_encoded, input_data_scaled], axis=1)

            # Make a prediction
            prediction = model.predict(input_data_prepared)
            prediction_proba = model.predict_proba(input_data_prepared)[0]

            # Return prediction result
            return render_template('result.html', prediction=prediction[0], probability=prediction_proba[1])

        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
