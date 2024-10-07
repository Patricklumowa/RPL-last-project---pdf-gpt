from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the Hugging Face model
model = pipeline("text-generation")

@app.route("/generate_text", methods=["POST"])
def generate_text():
    # Get the user input from the request
    user_input = request.json["user_input"]

    # Use the Hugging Face model to generate text
    generated_text = model(user_input, max_length=100)

    # Return the generated text as a JSON response
    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(debug=True)
