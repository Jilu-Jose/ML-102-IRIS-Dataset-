from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


model_path = 'iris_model.pkl'
model = joblib.load(model_path)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


app = Flask(__name__, template_folder='templates_one')  

@app.route('/')
def index():
    
    return render_template('index_one.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.json
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
      
        prediction = model.predict(features)[0]
        species = label_encoder.inverse_transform([prediction])[0]
        
        
        return jsonify({'prediction': species})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
