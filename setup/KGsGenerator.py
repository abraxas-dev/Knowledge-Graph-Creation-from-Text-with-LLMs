import os
from pathlib import Path
from transformers import pipeline
import torch

class KGsGeneratorWithPipeline:

    def __init__(self, input_dir, output_dir, pipe_type, model_name, max_chunk_length: int = 450, batch_size: int = 1):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

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
                device = 0 if torch.cuda.is_available() == "cuda" else -1,
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True
                }
            )
        except Exception as e:
            print(f"Failed to initialize pipeline : {str(e)}")
            raise
    def generate(self):
        pass


def main():

    input_dir = "Path"
    output_dir = "Path"
    model_name = "Name"
    max_chunk_length = 123
    batch_size = 1
    pipe_type = "Type"

    generator = KGsGeneratorWithPipeline(input_dir, output_dir, pipe_type, model_name, max_chunk_length, batch_size)

if __name__ == "__main__":
    main()