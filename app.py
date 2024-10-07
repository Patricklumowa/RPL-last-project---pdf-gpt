from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/generate_text", methods=["POST"])
def generate_text():
    # Get the user input from the request
    user_input = request.json["user_input"]

    # Use the Hugging Face API to generate text
    api_url = "https://api-inference.huggingface.co/models/t5-base"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    data = {"inputs": user_input, "options": {"max_length": 100}}
    response = requests.post(api_url, headers=headers, json=data)

    # Return the generated text as a JSON response
    return jsonify({"generated_text": response.json()["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True)
