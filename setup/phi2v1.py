import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="auto", 
    torch_dtype=torch.float16,
    offload_folder="offload",
    low_cpu_mem_usage=True,
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

our_text = """The Internet (or internet) is the global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies. The Internet carries a vast range of information resources and services, such as the interlinked hypertext documents and applications of the World Wide Web (WWW), electronic mail, telephony, and file sharing.
The origins of the Internet date back to research that enabled the time-sharing of computer resources, the development of packet switching in the 1960s and the design of computer networks for data communication. The set of rules (communication protocols) to enable internetworking on the Internet arose from research and development commissioned in the 1970s by the Defense Advanced Research Projects Agency (DARPA) of the United States Department of Defense in collaboration with universities and researchers across the United States and in the United Kingdom and France. The ARPANET initially served as a backbone for the interconnection of regional academic and military networks in the United States to enable resource sharing. The funding of the National Science Foundation Network as a new backbone in the 1980s, as well as private funding for other commercial extensions, encouraged worldwide participation in the development of new networking technologies and the merger of many networks using DARPA's Internet protocol suite. The linking of commercial networks and enterprises by the early 1990s, as well as the advent of the World Wide Web, marked the beginning of the transition to the modern Internet, and generated sustained exponential growth as generations of institutional, personal, and mobile computers were connected to the network. Although the Internet was widely used by academia in the 1980s, the subsequent commercialization in the 1990s and beyond incorporated its services and technologies into virtually every aspect of modern life.
Most traditional communication media, including telephone, radio, television, paper mail, and newspapers, are reshaped, redefined, or even bypassed by the Internet, giving birth to new services such as email, Internet telephone, Internet television, online music, digital newspapers, and video streaming websites. Newspapers, books, and other print publishing have adapted to website technology or have been reshaped into blogging, web feeds, and online news aggregators. The Internet has enabled and accelerated new forms of personal interaction through instant messaging, Internet forums, and social networking services. Online shopping has grown exponentially for major retailers, small businesses, and entrepreneurs, as it enables firms to extend their "brick and mortar" presence to serve a larger market or even sell goods and services entirely online. Business-to-business and financial services on the Internet affect supply chains across entire industries."""

messages = [
    {"role": "system", "content": "You are an AI assistant specialized in creating knowledge graphs."},
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

formatted_messages = "\n".join(
    [f"{message['role']}: {message['content']}" if isinstance(message['content'], str) else f"{message['role']}: {message['content']}" for message in messages]
)

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
output = pipe(formatted_messages, **generation_args)

print(output[0]['generated_text'])
