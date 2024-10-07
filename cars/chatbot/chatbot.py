# This file will contain the logic for the chatbot
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize the GODEL model and tokenizer
model_name = "microsoft/GODEL-v1_1-base-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_chatbot_response(user_message, knowledge="", history=""):
    # Prepare the input for GODEL
    prompt = f"Human: {user_message}\nAssistant: "
    if knowledge:
        prompt = f"Knowledge: {knowledge}\n" + prompt
    if history:
        prompt = f"Human: {history}\nHuman: {user_message}\nAssistant: "

    # Encode the input
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)
    
    # Generate a response
    outputs = model.generate(
        **inputs,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip()
