# This file will contain the logic for the chatbot
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize the GODEL model and tokenizer
model_name = "microsoft/GODEL-v1_1-base-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_chatbot_response(user_message, history=""):
    # Prepare the input for GODEL
    knowledge = """
Knowledge:
Basic car issues often revolve around common maintenance and mechanical problems that can impact the performance and safety of a vehicle. One of the most frequent issues is a dead battery, usually caused by leaving lights on or a faulty charging system. It's essential to check the battery regularly, especially in cold weather, and ensure that the terminals are clean and free from corrosion. Flat tires are another common problem, which may be due to punctures or worn-out treads. Maintaining the correct tire pressure and regularly inspecting for damage can help prevent this issue. 
Other frequent car problems include engine overheating, often due to low coolant levels, a faulty thermostat, or a broken water pump. Regularly checking fluid levels and ensuring proper cooling system maintenance can help avoid this. Brake wear is another crucial issue; if you hear squealing or grinding sounds, the brake pads may be worn and require replacement. Keeping an eye on your car's dashboard warning lights is key to identifying these issues early, as they often signal problems with the engine, oil, or other critical systems.
Electric car charging issues are commonly centered around the availability, speed, and compatibility of charging infrastructure. One major challenge is finding accessible charging stations, especially in areas with limited infrastructure. While urban areas tend to have more public charging points, rural or less-developed regions may not, causing range anxiety for drivers who worry about running out of charge before reaching a station. Installing home chargers can alleviate some of these concerns, but not all homes, especially apartments, may have the ability to install them.
Charging speed is another concern for electric vehicle (EV) owners. Charging at home or at slower public stations can take several hours, which might not be convenient for drivers who need a quick top-up. Fast-charging stations, which can charge an EV in 30 minutes or less, are helpful but not always widely available, and frequent use of these can also degrade the battery over time. Compatibility is another issue; not all charging stations support every type of EV connector, meaning that drivers need to plan their routes around compatible stations or carry multiple adapters. Regularly updating software in EVs can also help improve charging efficiency and solve some compatibility issues.
Some popular electric car models include the Tesla Model S, Tesla Model 3, Tesla Model X, Tesla Model Y, Nissan Leaf, Chevrolet Bolt EV, BMW i3, Audi e-tron, Ford Mustang Mach-E, Hyundai Kona Electric, Hyundai Ioniq Electric, Porsche Taycan, Volkswagen ID.4, Kia EV6, Kia Niro EV, Jaguar I-PACE, Rivian R1T, Lucid Air, Polestar 2, Volvo XC40 Recharge, Mercedes-Benz EQC, Mini Electric, Honda e, and BYD Tang EV.
Instructions:
You are a chatbot identified as System. You will be provided with a conversation history and a user message. You need to generate a response based on the conversation history and the user message (question). You will identify the type of car and the issue first. Don't repeat the history, take into accounts the information the user has already provided, avoid repeating the same information. You know the car models provided above in the knowledge section.
    """
    
    prompt = f"User: {user_message}\nSystem: "
    
    if history:
        history = '\n'.join(f"- {item["sender"]}: '{item["message"]}'" for item in history)
        prompt = f"History:\n{history}\n" + prompt
    
    prompt = f"Knowledge:\n{knowledge}\n" + prompt
    
    print(prompt)

    # Encode the input
    inputs = tokenizer([prompt[-1000:]], return_tensors="pt")
    print(f"Inputs: {inputs['input_ids'].shape}")
    
    # Generate a response
    outputs = model.generate(
        **inputs,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=250,
        top_p=0.99,
        temperature=0.99,
    )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip()
