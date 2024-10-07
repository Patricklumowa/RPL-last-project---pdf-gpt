from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the Hugging Face model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def generate_text(user_input):
    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Use the Hugging Face model to generate text
    outputs = model.generate(inputs["input_ids"], max_length=100)

    # Return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
