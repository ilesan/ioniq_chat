# This file will contain the logic for the chatbot
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_chatbot_response(user_message):
    # Encode the user's message and add it to the chat history
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    
    # Generate a response
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode the generated response
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response
