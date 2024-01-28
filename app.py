from flask import Flask, request, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import PyPDF2

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Extract text from PDF
pdf_path = '48lawsofpower.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Generate route
@app.route('/generate', methods=['POST'])
def generate():
    # Get user input question from the form
    question = request.form['question']
    
    # Generate answer using the GPT-2 model
    input_text = pdf_text + " " + question  # Combine PDF text with user question
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Render the template with the original question and generated answer
    return render_template('index.html', question=question, answer=generated_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
