import os
from pathlib import Path
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class KGsGeneratorWithModel:
    """
    A class for generating Knowledge Graph triples from text using a language model.
    Processes text files and generates structured knowledge representations.
    """

    def __init__(self, input_dir, output_dir, prompt_template, model_name, max_chunk_length: int = 450, batch_size: int = 1):
        """
        Initialize the KG Generator with specified parameters.
        
        Args:
            input_dir: Directory containing input text files
            output_dir: Directory where generated responses will be saved
            prompt_template: Template string for formatting prompts
            model_name: Name/path of the pretrained model to use
            max_chunk_length: Maximum length of text chunks to process
            batch_size: Number of chunks to process simultaneously
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size

        self._initialize_output_dir()
        self._initialize_model()
    
    def _initialize_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_model(self):
        """
        Initialize the language model and tokenizer.
        Sets up the model on GPU if available, otherwise on CPU.
        """
        try:
            # Determine device (GPU/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Initialize model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            self.device = device
        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            raise

    def generate_prompt(self, request):
        """
        Format the input text using the prompt template.
        
        Args:
            request: Input text to be formatted
        Returns:
            Formatted prompt string
        """
        try:
            return self.prompt_template.format(text=request)
        except Exception as e:
            print(f"Failed to generate a prompt: {str(e)}")
            raise
    
    def generate_response(self, text):
        """
        Generate a response using the language model.
        
        Args:
            text: Input text to process
        Returns:
            Generated response string
        """
        try:
            # Format prompt and tokenize
            prompt = self.generate_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response and remove prompt
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):]  # Remove the input prompt
            return response
        
        except Exception as e:
            print(f"Failed to generate a response: {str(e)}")
            raise

    def save_response(self, filename, response):
        """
        Save the generated response to a file.
        
        Args:
            filename: Name of the file to save (without extension)
            response: Generated response text to save
        """
        try:
            output_file = self.output_dir / f"{filename}_response.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Response saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save response: {str(e)}")
            raise

    def process_file(self, file_path):
        """
        Process a single input file and generate its response.
        
        Args:
            file_path: Path to the input file
        """
        try:
            print(f"Processing file: {file_path}")
            start_time = time.time()
            
            # Read input file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Generate and save response
            response = self.generate_response(text)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.save_response(file_path.stem, response)
            print(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Failed to process a file {file_path}: {str(e)}")

    def process(self):
        """
        Process all text files in the input directory.
        Shows progress bar and handles errors.
        """
        try:
            # Get list of text files
            txt_files = list(self.input_dir.glob("*.txt"))
            total_files = len(txt_files)

            if total_files == 0:
                print("No .txt files found in the input directory")
                return

            # Initialize progress bar
            progress_bar = tqdm(
                total=total_files,
                desc="Summarizing emails",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

            # Process each file
            for file_path in txt_files:
                self.process_file(file_path)

        except Exception as e:
            print(f"Failed to process : {str(e)}")
            raise

def main():
    """
    Main entry point of the script.
    Sets up configuration and runs the KG generation process.
    """
    input_dir = ""  # Directory containing input text files
    output_dir = ""  # Directory where responses will be saved
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Model to use
    max_chunk_length = 123  # Maximum length of text chunks
    batch_size = 1  # Number of chunks to process at once
    prompt_template = """
    Generate Triples for the following text:
    {text}
    """

    # Initialize and run the generator
    generator = KGsGeneratorWithModel(input_dir, output_dir, prompt_template, model_name, max_chunk_length, batch_size)
    generator.process()

if __name__ == "__main__":
    main()
