import os
from pathlib import Path
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class TripleGenerator:
    """
    A class for generating Knowledge Graph triples from text using a language model.
    Processes text files and generates structured knowledge representations.
    """
    def __init__(self, api_key, input_dir, output_dir, system_message, prompt_template, model_name, temperature, max_new_tokens: int = 450, batch_size: int = 1):
        """
        Initialize the KG Generator with specified parameters.
        """
        self.api_key = api_key
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature

        self._initialize_output_dir()
        self._initialize_model()
    
    def _initialize_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_model(self):
        """
        Initialize the language model and tokenizer.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            self.device = device
        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            raise

    def generate_prompt(self, request):
        """
        Format the input text using system message and user prompt template.
        """
        try:
            formatted_prompt = f"""{self.system_message}

            {self.prompt_template.format(text=request)}"""
            return formatted_prompt
        except Exception as e:
            print(f"Failed to generate a prompt: {str(e)}")
            raise
    
    def generate_response(self, text):
        """
        Generate a response using the language model.
        """
        try:
            prompt = self.generate_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):]  # Remove the input prompt
            return response
        except Exception as e:
            print(f"Failed to generate a response: {str(e)}")
            raise

    def parse_and_save_triples(self, response, output_file):
        """
        Parse the response text into formatted triples and write each directly to the file.
        """
        try:
            lines = response.strip().split("\n")
            with open(output_file, 'a', encoding='utf-8') as f:
                for line in lines:
                    line = line.lstrip("0123456789. ")  # Remove numbering (e.g., "1. ")
                    parts = line.strip("()").split(", ")
                    if len(parts) == 3:
                        # Clean formatting for each part of the triple and write directly to file
                        formatted_triple = f'("{parts[0].strip()}", "{parts[1].strip()}", "{parts[2].strip()}");'
                        f.write(formatted_triple + "\n")
        except Exception as e:
            print(f"Failed to parse and save triples: {str(e)}")
            raise

    def save_response_to_file(self, response, file_path):
        """
        Save the raw LLM response to a file.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Raw LLM response saved to {file_path}")
        except Exception as e:
            print(f"Failed to save LLM response to file: {str(e)}")
            raise

    def process_file(self, file_path, triples_file):
        """
        Process a single input file and directly save its triples to a file.
        """
        try:
            print(f"Processing file: {file_path}")
            start_time = time.time()
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            response = self.generate_response(text)
            
            # Save the raw response to a file
            raw_response_file = self.output_dir / f"{Path(file_path).stem}_response.txt"
            self.save_response_to_file(response, raw_response_file)
            
            # Parse and save triples directly to the triples file
            self.parse_and_save_triples(response, triples_file)
            end_time = time.time()
            print(f"Successfully processed {file_path} in {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Failed to process file {file_path}: {str(e)}")

    def process(self):
        """
        Process all text files in the input directory and save triples directly to a file.
        """
        try:
            txt_files = list(self.input_dir.glob("*.txt"))
            if not txt_files:
                print("No .txt files found in the input directory.")
                return
            
            # Output file for triples
            triples_file = self.output_dir / "generated_triples.txt"
            
            # Ensure the triples file is empty before writing
            open(triples_file, 'w').close()

            for file_path in tqdm.tqdm(txt_files, desc="Processing files", unit="file"):
                self.process_file(file_path, triples_file)

            print(f"All triples saved to {triples_file}")
        except Exception as e:
            print(f"Failed to process files: {str(e)}")
            raise


if __name__ == "__main__":
    input_dir = "text"
    output_dir = "./wtv"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens = 300
    batch_size = 1
    temperature = 0.1  # Ensure this is > 0
    system_message = """
    Extract RDF triples from the following text. Each triple should be of the form (subject, predicate, object).

    Example:
    Text: "The Eiffel Tower is located in Paris, France, and was completed in 1889."
    Output:
    1. (Eiffel Tower, is located in, Paris)
    2. (Paris, is in, France)
    3. (Eiffel Tower, was completed in, 1889)
    """
    prompt_template = """
    Generate Triples for the following text:
    {text}
    """
    generator = TripleGenerator("", input_dir, output_dir, system_message, prompt_template, model_name, temperature, max_new_tokens, batch_size)
    generator.process()
