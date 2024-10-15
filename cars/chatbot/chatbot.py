# This file will contain the logic for the chatbot
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize the GODEL model and tokenizer
model_name = "microsoft/GODEL-v1_1-large-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_chatbot_response(user_message, knowledge="", history=[]):
    instruction = f"Instruction: given a dialog context, you need to response empathically. You are an electric car expert. If the car model is not known, ask if it's an Ioniq. If it's an Ioniq, propose the most likely explanation of the issue with the car, one issue at a time. Do NOT just give the solution, propose one issue at a time. Do NOT just return the knowledge, propose one issue at a time. Reply in full sentences."
    
    dialog = ' EOS '.join([f"{h['sender']}: {h['message']}" for h in history] + [f"User: {user_message}"])

    prompt = f"{instruction} [CONTEXT] {dialog}"

    if knowledge:
        prompt = f"{prompt} [KNOWLEDGE] {knowledge}"
    
    # print(prompt)

    # Encode the input with truncation
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # print(f"Inputs: {inputs}")
    
    # Generate a response
    outputs = model.generate(
        inputs,  # Correctly pass the input_ids
        max_length=100,       # Adjust max_length to a reasonable value
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=10,             # Adjust top_k for better sampling
        top_p=0.95,           # Adjust top_p for better sampling
        temperature=0.5,      # Adjust temperature for more diverse responses
    )
    # return outputs[0]
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip()
