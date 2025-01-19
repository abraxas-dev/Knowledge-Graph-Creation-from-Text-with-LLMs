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
    def __init__(self, api_key, input_dir, output_dir, system_message, prompt_template, model_name, temperature, max_new_tokens: int = 450, batch_size: int = 1, max_chunks: int = None):
        """
        Initialize the KG Generator with specified parameters.
        
        Args:
            input_dir: Directory containing input text files
            output_dir: Directory where generated responses will be saved
            prompt_template: Template string for formatting prompts
            model_name: Name/path of the pretrained model to use
            max_new_tokens: Maximum length of text chunks to process
            batch_size: Number of chunks to process simultaneously
            max_chunks: Maximum number of chunks to process per subdirectory (None for no limit)
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
        self.max_chunks = max_chunks

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
            print("\n" + "="*50)
            print("🔄 Initializing Triple Generator Model...")
            print("="*50)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📍 Using device: {self.device}")
            
            print("🔄 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print("🔄 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            
            print("✅ Model initialization complete!")
            print("="*50 + "\n")
            
        except Exception as e:
            print("❌ Model initialization failed!")
            raise

    def generate_prompt(self, request):
        """
        Format the input text using system message and user prompt template.
        
        Args:
            text: Input text to be formatted
        Returns:
            Formatted prompt string with system message and user input
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
        
        Args:
            text: Input text to process
        Returns:
            Generated response string
        """
        try:
            # Format prompt and tokenize
            prompt = self.generate_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response with gradients disabled for inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=self.temperature,
                    max_new_tokens = self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response and remove prompt
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

    def save_response(self, filename, response, input_dir=None):
        """
        Save the generated response to a file, maintaining directory structure.
        
        Args:
            filename: Name of the file to save (without extension)
            response: Generated response text to save
            input_dir: Original input directory path to maintain structure
        """
        try:
            # If input_dir is provided, create corresponding subdirectory in output
            if input_dir:
                # Get the relative path from input_dir to the processed directory
                rel_path = Path(input_dir).relative_to(self.input_dir)
                output_subdir = self.output_dir / rel_path
                os.makedirs(output_subdir, exist_ok=True)
                
                output_file = output_subdir / f"{filename}_response.txt"
                triples_file = output_subdir / f"{filename}_triples.txt"
            else:
                output_file = self.output_dir / f"{filename}_response.txt"
                triples_file = self.output_dir / f"{filename}_triples.txt"

            # First save the raw response
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Raw response saved to: {output_file}")
            
            # Then parse and save formatted triples
            self.parse_and_save_triples(response, triples_file)
            print(f"Formatted triples saved to: {triples_file}")
        except Exception as e:
            print(f"Failed to save response: {str(e)}")
            raise

    def process_file(self, file_path):
        """
        Process a single input file and generate its response.
        
        Args:
            file_path: Path to the input file
        Returns:
            float: Time taken to process the file
        """
        try:
            print(f"\n Processing file: {file_path}")
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print("🔄 Generating response...")
            response = self.generate_response(text)
            
            # Pass the parent directory to maintain structure
            self.save_response(file_path.stem, response, input_dir=file_path.parent)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"✅ File processed in {elapsed_time:.2f} seconds")
            print(f" Response saved for {file_path.stem}")
            return elapsed_time
            
        except Exception as e:
            print(f"❌ Failed to process file {file_path}")
            raise

    def process_directory(self, directory):
        """
        Process all text files in a specific directory.
        
        Args:
            directory: Path object pointing to the directory to process
        Returns:
            tuple: (total_files_time, total_dir_time)
        """
        try:
            dir_start_time = time.time()
            txt_files = list(directory.glob("*.txt"))
            if not txt_files:
                print(f"No .txt files found in {directory}")
                return 0, 0

            print(f"\n📂 Processing directory: {directory.name}")
            
            # Sort files by chunk number
            txt_files.sort(key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[0] == 'chunk' else float('inf'))
            
            # Apply max_chunks limit if specified
            if self.max_chunks is not None and len(txt_files) > self.max_chunks:
                print(f"⚠️  Found {len(txt_files)} files, limiting to first {self.max_chunks} chunks as per max_chunks setting")
                txt_files = txt_files[:self.max_chunks]
            
            print(f"📄 Processing {len(txt_files)} files")
            
            total_files_time = 0
            for file_path in txt_files:
                file_time = self.process_file(file_path)
                total_files_time += file_time
            
            dir_end_time = time.time()
            total_dir_time = dir_end_time - dir_start_time
            
            print(f"\n📊 Directory Statistics for {directory.name}:")
            print(f"   ⏱️  Total files processing time: {total_files_time:.2f} seconds")
            print(f"   ⏱️  Total directory time (including overhead): {total_dir_time:.2f} seconds")
            print(f"   ⏱️  Overhead time: {(total_dir_time - total_files_time):.2f} seconds")
            print(f"✅ Completed processing directory: {directory.name}")
            
            return total_files_time, total_dir_time

        except Exception as e:
            print(f"Failed to process directory {directory}: {str(e)}")
            raise

    def process(self):
        """
        Process files in the input directory. If there are subdirectories, process those.
        Otherwise, process .txt files directly in the input directory.
        Shows progress bar and handles errors.
        """
        try:
            # Check for subdirectories first
            subdirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
            
            if subdirs:
                # Process subdirectories
                print(f"Found {len(subdirs)} subdirectories to process")
                progress_bar = tqdm.tqdm(
                    total=len(subdirs),
                    desc="Processing subdirectories",
                    unit="dir",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                total_processing_time = 0
                total_overhead_time = 0
                for subdir in subdirs:
                    files_time, dir_time = self.process_directory(subdir)
                    total_processing_time += files_time
                    total_overhead_time += (dir_time - files_time)
                    progress_bar.update(1)

                progress_bar.close()
                print("\n📊 Overall Statistics:")
                print(f"   ⏱️  Total files processing time: {total_processing_time:.2f} seconds")
                print(f"   ⏱️  Total overhead time: {total_overhead_time:.2f} seconds")
                print(f"   ⏱️  Total execution time: {(total_processing_time + total_overhead_time):.2f} seconds")
            else:
                # Process files directly in input directory
                txt_files = list(self.input_dir.glob("*.txt"))
                if not txt_files:
                    print("No .txt files found in the input directory")
                    return

                print(f"Processing {len(txt_files)} files in root directory")
                files_time, total_time = self.process_directory(self.input_dir)
                print("\n📊 Root Directory Statistics:")
                print(f"   ⏱️  Total processing time: {files_time:.2f} seconds")
                print(f"   ⏱️  Total overhead time: {(total_time - files_time):.2f} seconds")

        except Exception as e:
            print(f"Failed to process: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Main entry point of the script.
    Sets up configuration and runs the KG generation process.
    """
    input_dir = "/Users/abraxas/Desktop/Desktop/Studium/7 Semester/Data Engineering/src/test_processed_data/en.wikipedia.org_Artificial_intelligence_-_Wikipedia"  # Directory containing input text files
    output_dir = "./wtv"  # Directory where responses will be saved
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Model to use
    max_new_tokens = 400  # Maximum length of text chunks
    batch_size = 1  # Number of chunks to process at once
    temperature = 0
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

    # Initialize and run the generator
    generator = TripleGenerator("", input_dir, output_dir, system_message, prompt_template, model_name, temperature, max_new_tokens, batch_size)
    generator.process()
