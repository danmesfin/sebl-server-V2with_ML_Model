from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

# Load the TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path='/assets/yield_prediction_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

# Global variables for the model
interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        data = request.json
        parameters = data['parameters']

        # Convert input data to a numpy array and reshape
        input_data = np.array(parameters, dtype=np.float32).reshape((1, 115))

        # Perform prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process the prediction result
        predicted_yield = output_data[0][0]

        # Return the predicted result as a JSON response
        response = {'predicted_yield': predicted_yield}

        return jsonify(response)

@app.route("/")
def hello():
    return "Hello world!"

if __name__ == '__main__':
    app.run()
