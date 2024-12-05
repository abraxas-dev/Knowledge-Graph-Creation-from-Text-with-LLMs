import os
from pathlib import Path
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class KGsGeneratorWithModel:

    def __init__(self, input_dir, output_dir, prompt_template, model_name, max_chunk_length: int = 450, batch_size: int = 1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size

        self._initialize_output_dir()
        self._initialize_model()
    
    def _initialize_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_model(self):
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
        try:
            return self.prompt_template.format(text=request)
        except Exception as e:
            print(f"Failed to generate a prompt: {str(e)}")
            raise
    
    def generate_response(self, text):
        try:
            prompt = self.generate_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(prompt):]
            return response
        
        except Exception as e:
            print(f"Failed to generate a response: {str(e)}")
            raise

    def save_response(self, filename, response):
        try:
            output_file = self.output_dir / f"{filename}_response.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Response saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save response: {str(e)}")
            raise

    def process_file(self, file_path):
        try:
            print(f"Processing file: {file_path}")
            start_time = time.time()
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            response = self.generate_response(text)
            end_time = time.time()  # End time measurement
            elapsed_time = end_time - start_time
            self.save_response(file_path.stem, response)
            print(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Failed to process a file {file_path}: {str(e)}")

    def process(self):
        try:
            txt_files = list(self.input_dir.glob("*.txt"))
            total_files = len(txt_files)

            if total_files == 0:
                print("No .txt files found in the input directory")
                return

            progress_bar = tqdm(
                total=total_files,
                desc="Summarizing emails",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

            successful = 0
            failed = 0

            for file_path in txt_files:
                self.process_file(file_path)

        except Exception as e:
            print(f"Failed to process : {str(e)}")
            raise

def main():
    input_dir = ""
    output_dir = ""
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Updated model name
    max_chunk_length = 123
    batch_size = 1
    prompt_template = """
    Generate Triples for the following text:
    {text}
    """

    generator = KGsGeneratorWithModel(input_dir, output_dir, prompt_template, model_name, max_chunk_length, batch_size)
    generator.process()

if __name__ == "__main__":
    main()
