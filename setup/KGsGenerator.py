import os
from pathlib import Path
from transformers import pipeline
import torch

class KGsGeneratorWithPipeline:

    def __init__(self, input_dir, output_dir, prompt_template, pipe_type, model_name, max_chunk_length: int = 450, batch_size: int = 1):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prompt_template = prompt_template
        self.pipe_type = pipe_type
        self.model_name = model_name
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size

        self._initialize_output_dir()
        self._initialize_pipe()
    
    def _initialize_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_pipe(self):
        try:
            self.pipe = pipeline(
                self.pipe_type,
                model = self.model_name,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                }
            )
        except Exception as e:
            print(f"Failed to initialize pipeline : {str(e)}")
            raise

    def generate_prompt(self, request):
        try:
            return self.prompt_template.format(text=request)
        except Exception as e:
            print(f"Failed to generate a promt : {str(e)}")
            raise
    
    def generate_response(self, text):
        try:
            request = self.generate_prompt(text)
            response = self.pipe(request, max_new_tokens=400)
            return response[0]['generated_text']
        
        except Exception as e:
            print(f"Failed to generate a response : {str(e)}")
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
            for file_path in txt_files:
                self.process_file(file_path)

        except Exception as e:
            print(f"Failed to process : {str(e)}")
            raise

def main():

    input_dir = ""
    output_dir = ""
    model_name = "microsoft/Phi-3.5-mini-instruct"
    max_chunk_length = 123
    batch_size = 1
    pipe_type = "text-generation"
    prompt_template = """
    Generate Triples for the following text:
    {text}
    """

    generator = KGsGeneratorWithPipeline(input_dir, output_dir, prompt_template, pipe_type, model_name, max_chunk_length, batch_size)
    generator.process()

if __name__ == "__main__":
    main()
