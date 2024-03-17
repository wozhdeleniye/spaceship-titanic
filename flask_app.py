from flask import Flask, request, jsonify
from model import My_Classifier_Model

app = Flask(__name__)
model = My_Classifier_Model()

@app.route('/train', methods=['POST'])
def train_model():
    model.train()
    return jsonify({'message': 'Model training completed.'})

@app.route('/predict', methods=['POST'])
def predict_model():
    model.prediction()
    return jsonify({'message': 'Prediction completed. Result saved in ./data/result.csv'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)