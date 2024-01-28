import streamlit as st
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

# Streamlit app
def main():
    st.title('GPT-2 Text Generation')
    question = st.text_input('Enter your question:')
    if st.button('Generate Answer'):
        input_text = pdf_text + " " + question  # Combine PDF text with user question
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.text_area('Generated Answer:', value=generated_text, height=200)

if __name__ == '__main__':
    main()
