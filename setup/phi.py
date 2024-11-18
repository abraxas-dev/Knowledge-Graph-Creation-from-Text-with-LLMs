import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

our_text = "The Internet is a global network of interconnected computers that enables communication and data sharing worldwide. It was initially developed in the 1960s as ARPANET by the United States Department of Defense. Over time, it evolved into a public network accessible to individuals, businesses, and governments. The Internet relies on technologies like the World Wide Web, which allows access to websites and information through URLs, and email, a system for exchanging messages. Key protocols such as TCP/IP ensure reliable data transfer between devices. Today, the Internet is fundamental to modern life, supporting activities like online shopping, social networking, and streaming media."
messages = [
    {"role": "system", "content": "You are an AI assistant specialized in creating knowledge graphs."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "user", "content": "Create a knowledge graph from the following text: 'The Earth is the third planet in the solar system. It orbits the Sun in an elliptical path and has a natural satellite, the Moon. The Earth supports life thanks to its atmosphere and the presence of water.'"},
    {
        "role": "assistant",
        "content": {
            "nodes": [
                {"id": "Earth", "type": "Planet"},
                {"id": "Sun", "type": "Star"},
                {"id": "Moon", "type": "Natural Satellite"},
                {"id": "Water", "type": "Substance"},
                {"id": "Atmosphere", "type": "Layer"},
                {"id": "Life", "type": "State"}
            ],
            "edges": [
                {"source": "Earth", "relation": "is", "target": "Planet"},
                {"source": "Earth", "relation": "orbits", "target": "Sun"},
                {"source": "Earth", "relation": "has", "target": "Moon"},
                {"source": "Earth", "relation": "supports", "target": "Life"},
                {"source": "Earth", "relation": "thanks to", "target": "Atmosphere"},
                {"source": "Earth", "relation": "thanks to", "target": "Water"}
            ]
        }
    },
    {"role": "user", "content": our_text},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
