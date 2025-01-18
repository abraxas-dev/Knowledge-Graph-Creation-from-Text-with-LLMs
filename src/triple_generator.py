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
    def __init__(self, api_key, input_dir, output_dir, system_message, prompt_template, model_name, temperature, max_new_tokens: int = 450, batch_size: int = 1, unified_output_file="all_generated_triples.txt"):
        """
        Initialize the KG Generator with specified parameters.
        
        Args:
            unified_output_file: Name of the single file to store all generated triples
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
        self.unified_output_file = self.output_dir / unified_output_file
        
        # Files for valid and invalid triples
        self.valid_triples_file = self.output_dir / "valid_triples.txt"
        self.invalid_triples_file = self.output_dir / "invalid_triples.txt"

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
            request: Input text to be formatted
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
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response and remove prompt
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):]  # Remove the input prompt
            return response
        
        except Exception as e:
            print(f"Failed to generate a response: {str(e)}")
            raise

    def parse_and_save_triples(self, response, output_file, append=False):
        """
        Parse the response text into formatted triples and write them to a file.
        
        Args:
            response: The raw response text from the LLM
            output_file: The file to save the triples
            append: If True, append to the file. If False, overwrite the file.
        """
        try:
            mode = 'a' if append else 'w'
            lines = response.strip().split("\n")
            with open(output_file, mode, encoding='utf-8') as f:
                for line in lines:
                    line = line.lstrip("0123456789. ")  # Remove numbering (e.g., "1. ")
                    parts = line.strip("()").split(", ")
                    if len(parts) == 3:
                        # Clean formatting for each part of the triple
                        formatted_triple = f'("{parts[0].strip()}", "{parts[1].strip()}", "{parts[2].strip()}");'
                        f.write(formatted_triple + "\n")
        except Exception as e:
            print(f"Failed to parse and save triples: {str(e)}")
            raise

    def validate_triples(self):
        """
        Validate triples from the unified output file and separate them into valid and invalid files.
        """
        try:
            with open(self.unified_output_file, 'r', encoding='utf-8') as input_file, \
                 open(self.valid_triples_file, 'w', encoding='utf-8') as valid_file, \
                 open(self.invalid_triples_file, 'w', encoding='utf-8') as invalid_file:

                for line in input_file:
                    line = line.strip()
                    if line.startswith("(") and line.endswith(");"):  # Basic triple format check
                        parts = line.strip("();").split('", "')
                        if len(parts) == 3 and all(parts):
                            valid_file.write(line + "\n")  # Write valid triple
                        else:
                            invalid_file.write(line + "\n")  # Write invalid triple
                    else:
                        invalid_file.write(line + "\n")  # Write invalid triple
            print(f"Validation complete. Valid triples saved to: {self.valid_triples_file}")
            print(f"Invalid triples saved to: {self.invalid_triples_file}")
        except Exception as e:
            print(f"Failed to validate triples: {str(e)}")
            raise

    def save_response(self, filename, response):
        """
        Save the generated response to a file and append all triples to a unified file.
        
        Args:
            filename: Name of the file to save (without extension)
            response: Generated response text to save
        """
        try:
            raw_output_file = self.output_dir / f"{filename}_response.txt"
            # Save the raw response
            with open(raw_output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Raw response saved to: {raw_output_file}")
            
            # Parse and append all triples to the unified output file
            self.parse_and_save_triples(response, self.unified_output_file, append=True)
            print(f"All triples appended to: {self.unified_output_file}")
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
            print(f"\n Processing file: {file_path}")
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print("🔄 Generating response...")
            response = self.generate_response(text)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.save_response(file_path.stem, response)
            print(f"✅ File processed in {elapsed_time:.2f} seconds")
            print(f" Response saved for {file_path.stem}")
            
        except Exception as e:
            print(f"❌ Failed to process file {file_path}")
            raise

    def process_directory(self, directory):
        """
        Process all text files in a specific directory.
        """
        try:
            txt_files = list(directory.glob("*.txt"))
            if not txt_files:
                print(f"No .txt files found in {directory}")
                return

            print(f"\nProcessing directory: {directory.name}")
            for file_path in txt_files:
                self.process_file(file_path)
            print(f"Completed processing directory: {directory.name}")

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

                for subdir in subdirs:
                    self.process_directory(subdir)
                    progress_bar.update(1)

                progress_bar.close()
            else:
                # Process files directly in input directory
                txt_files = list(self.input_dir.glob("*.txt"))
                if not txt_files:
                    print("No .txt files found in the input directory")
                    return

                print(f"Processing {len(txt_files)} files in root directory")
                progress_bar = tqdm.tqdm(
                    total=len(txt_files),
                    desc="Processing files",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                for file_path in txt_files:
                    self.process_file(file_path)
                    progress_bar.update(1)

                progress_bar.close()

            # Validate all triples after processing
            print("\n🔍 Validating all triples...")
            self.validate_triples()

        except Exception as e:
            print(f"Failed to process: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Main entry point of the script.
    Sets up configuration and runs the KG generation process.
    """
    input_dir = "test"
    output_dir = "output"
    model_name = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens = 400
    batch_size = 1
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

    generator = TripleGenerator(
        "", 
        input_dir, 
        output_dir, 
        system_message, 
        prompt_template, 
        model_name, 
        temperature, 
        max_new_tokens, 
        batch_size
    )
    generator.process()

