"""
Project : Knowledge Graph Creation from Text
Author : @abraxas-dev
"""
import os
from pathlib import Path
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Dict
from src.utils.logger_config import setup_logger

class TripleGenerator:
    """
    A class for generating Knowledge Graph triples from text using a language model.
    Processes text files and generates structured knowledge representations.
    """
    def __init__(self, api_key: str, input_dir: str, output_dir: str, system_message: str, prompt_template: str, model_name: str, max_chunks: int = None, model_generate_parameters: Dict = None):
        """
        Initialize the TripleGenerator.
        
        Args:
            api_key: API key for the language model
            input_dir: Directory containing input text files
            output_dir: Directory to save generated triples
            system_message: System message for the LLM
            prompt_template: Template for formatting user prompts
            model_name: Name of the language model to use
            max_chunks: Maximum number of chunks to process (optional)
            model_generate_parameters: Model generation parameters (optional)
        """
        self.logger = setup_logger(__name__)
        self.api_key = api_key
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.max_chunks = max_chunks

        self.model_generate_parameters = {
            "temperature": 0.1,
            "max_new_tokens": 512,
            "pad_token_id": None  # Will be set after tokenizer initialization
        }
        
        # Update with any additional parameters from config
        if model_generate_parameters:
            self.model_generate_parameters = model_generate_parameters
        
        self._initialize_output_dir()
        self._initialize_model()
        self.model_generate_parameters["pad_token_id"] = self.tokenizer.eos_token_id
    
    def _initialize_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_model(self):
        """
        Initialize the language model and tokenizer.
        Sets up the model on GPU if available, otherwise on CPU.
        """
        try:
            self.logger.info("="*50)
            self.logger.info("🔄 Initializing Triple Generator Model...")
            self.logger.info("="*50)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"📍 Using device: {self.device}")
            
            self.logger.info("🔄 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            ## Which model to use ?
            ## torch.float - For more Storage, less accuracy
            ## toch.bfloat - Middle
            ## torch.float32 - the best accuracy, but also the most storage

            self.logger.info("🔄 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            
            self.logger.info("✅ Model initialization complete!")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error("❌ Model initialization failed!")
            raise

    def _generate_prompt(self, request):
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
            self.logger.error(f"Failed to generate a prompt: {str(e)}")
            raise
    
    def _generate_response(self, text):
        """
        Generate a response using the language model.
        
        Args:
            text: Input text to process
        Returns:
            Generated response string
        """
        try:
            # Format prompt and tokenize
            prompt = self._generate_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response with gradients disabled for inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.model_generate_parameters
                )
            
            # Decode response and remove prompt
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):] 
            return response
        
        except Exception as e:
            self.logger.error(f"Failed to generate a response: {str(e)}")
            raise

    def _parse_and_save_triples(self, response, output_file):
        """
        Parse the response text into formatted triples and write each directly to the file.
        Author : @paza15
        """
        try:
            lines = response.strip().split("\n")
            with open(output_file, 'a', encoding='utf-8') as f:
                for line in lines:
                    line = line.lstrip("0123456789. ")
                    parts = line.strip("()").split(", ")
                    if len(parts) == 3:
                        formatted_triple = f'("{parts[0].strip()}", "{parts[1].strip()}", "{parts[2].strip()}");'
                        f.write(formatted_triple + "\n")
        except Exception as e:
            self.logger.error(f"Failed to parse and save triples: {str(e)}")
            raise

    def _save_response(self, filename, response, input_dir=None):
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
            self.logger.info(f"✅ Raw response saved")
            
            # Then parse and save formatted triples
            self._parse_and_save_triples(response, triples_file)
            self.logger.info(f"✅ Formatted triples saved")
        except Exception as e:
            self.logger.error(f"Failed to save response: {str(e)}")
            raise

    def _process_file(self, file_path):
        """
        Process a single input file and generate its response.
        
        Args:
            file_path: Path to the input file
        Returns:
            float: Time taken to process the file
        """
        try:
            self.logger.info(f"🔄 Processing file: {file_path.stem}")
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.logger.info("🔄 Generating response...")
            response = self._generate_response(text)
            
            # Pass the parent directory to maintain structure
            self._save_response(file_path.stem, response, input_dir=file_path.parent)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.logger.info(f"⏱️  File processed in {elapsed_time:.2f} seconds")
            return elapsed_time
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process file {file_path}")
            raise
    
    def validate_triples(self):
        """
        Validate and deduplicate triples from all `_triples.txt` files in the output directory.
        Only keeps valid triples and ensures no duplicates are added.
        Author : @paza15
        """
        try:
            # Find all triple files
            response_files = list(self.output_dir.rglob("*_triples.txt"))
            if not response_files:
                self.logger.warning("No response files found for validation.")
                return

            all_valid_file = self.output_dir / "all_valid.txt"

            # Load existing triples to avoid duplicates
            existing_valid_triples = set()
            if all_valid_file.exists():
                with open(all_valid_file, 'r', encoding='utf-8') as f:
                    existing_valid_triples = {line.strip() for line in f if line.strip()}

            # Track statistics
            stats = {
                'total_processed': 0,
                'valid_new': 0,
                'duplicates': 0
            }

            # Process each response file
            for response_file in response_files:
                self.logger.info(f"Processing {response_file.name}...")
                
                with open(response_file, 'r', encoding='utf-8') as input_file:
                    for line in input_file:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        stats['total_processed'] += 1
                        
                        # Normalize the triple format
                        try:
                            # Remove trailing semicolon if present
                            if line.endswith(';'):
                                line = line[:-1]
                            
                            # Remove outer parentheses
                            line = line.strip('()')
                            
                            # Split parts and handle different quote styles
                            parts = []
                            current = []
                            in_quotes = False
                            quote_char = None
                            
                            for char in line:
                                if char in ['"', "'"]:
                                    if not in_quotes:
                                        in_quotes = True
                                        quote_char = char
                                    elif char == quote_char:
                                        in_quotes = False
                                        quote_char = None
                                elif char == ',' and not in_quotes:
                                    parts.append(''.join(current).strip())
                                    current = []
                                    continue
                                current.append(char)
                            
                            if current:
                                parts.append(''.join(current).strip())
                            
                            # Clean up parts
                            parts = [p.strip().strip('"\'') for p in parts]
                            
                            # Validate triple
                            is_valid = (
                                len(parts) == 3 and
                                all(len(part.strip()) > 0 for part in parts)
                            )

                            if is_valid:
                                # Format triple consistently
                                formatted_triple = f'("{parts[0]}", "{parts[1]}", "{parts[2]}");'
                                
                                if formatted_triple in existing_valid_triples:
                                    stats['duplicates'] += 1
                                    continue
                                
                                # Add new valid triple
                                existing_valid_triples.add(formatted_triple)
                                with open(all_valid_file, 'a', encoding='utf-8') as valid_f:
                                    valid_f.write(formatted_triple + "\n")
                                stats['valid_new'] += 1
                                self.logger.debug(f"Valid triple added: {formatted_triple}")
                                
                        except Exception as e:
                            self.logger.warning(f"Error parsing triple: {line} - {str(e)}")

                self.logger.info(f"Completed processing {response_file.name}")

            # Log statistics
            self.logger.info("="*50)
            self.logger.info("Validation Statistics:")
            self.logger.info(f"Total triples processed: {stats['total_processed']}")
            self.logger.info(f"New valid triples added: {stats['valid_new']}")
            self.logger.info(f"Duplicates skipped: {stats['duplicates']}")
            self.logger.info(f"Total valid triples: {len(existing_valid_triples)}")
            self.logger.info("="*50)
            self.logger.info(f"Valid triples file: {all_valid_file}")

            # Return statistics for potential use by caller
            return stats

        except Exception as e:
            self.logger.error(f"Failed to validate triples: {str(e)}")
            raise

    def _process_directory(self, directory):
        """
        Process all text files in a specific directory.
        
        Args:
            directory: Path object pointing to the directory to process
        Returns:
            float: Total processing time
        """
        try:
            txt_files = list(directory.glob("*.txt"))
            if not txt_files:
                self.logger.warning(f"No .txt files found in {directory}")
                return 0

            self.logger.info(f"📂 Processing directory: {directory.name}")
            
            # Sort files by chunk number
            txt_files.sort(key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[0] == 'chunk' else float('inf'))
            
            # Apply max_chunks limit if specified
            if self.max_chunks is not None and len(txt_files) > self.max_chunks:
                self.logger.info(f"⚠️  Found {len(txt_files)} files, limiting to first {self.max_chunks} chunks as per max_chunks setting")
                txt_files = txt_files[:self.max_chunks]
            
            self.logger.info(f"📄 Processing {len(txt_files)} files")
            
            total_processing_time = 0
            for file_path in txt_files:
                file_time = self._process_file(file_path)
                total_processing_time += file_time
            
            self.logger.info(f"📊 Directory Statistics for {directory.name}:")
            self.logger.info(f"     ⏱️  Total processing time: {total_processing_time:.2f} seconds")
            self.logger.info(f"     ✅ Completed processing directory: {directory.name}")

            #self.validate_triples()
            
            return total_processing_time

        except Exception as e:
            self.logger.error(f"Failed to process directory {directory}: {str(e)}")
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
                self.logger.info(f"📂 Found {len(subdirs)} subdirectories to process")
                progress_bar = tqdm.tqdm(
                    total=len(subdirs),
                    desc="Processing subdirectories",
                    unit="dir",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                total_processing_time = 0
                
                for subdir in subdirs:
                    processing_time = self._process_directory(subdir)
                    total_processing_time += processing_time
                    progress_bar.update(1)

                progress_bar.close()
                
                self.logger.info("📊 Overall Statistics:")
                self.logger.info(f"   ⏱️  Total processing time: {total_processing_time:.2f} seconds")
            else:
                # Process files directly in input directory
                txt_files = list(self.input_dir.glob("*.txt"))
                if not txt_files:
                    self.logger.warning("No .txt files found in the input directory")
                    return

                self.logger.info(f"Processing {len(txt_files)} files in root directory")
                processing_time = self._process_directory(self.input_dir)
                self.logger.info("📊 Root Directory Statistics:")
                self.logger.info(f"     ⏱️ Total processing time: {processing_time:.2f} seconds")
                self.logger.info(f"     ✅ Completed processing root directory")
            
            #self.logger.info("🔄 Validating generated triples...")
            #self.validate_triples()
            #self.logger.info("✅ Validation complete.")
        except Exception as e:
            self.logger.error(f"Failed to process: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Main entry point of the script.
    Sets up configuration and runs the KG generation process.
    """
    # Define all parameters directly
    input_dir = "../data/processed"  # Directory containing input text files
    output_dir = "./output_model"  # Directory where responses will be saved
    model_name = "microsoft/Phi-3.5-mini-instruct"  # Model to use
    
    # System message for triple extraction
    system_message = """
    Extract RDF triples from the following text. Each triple should be of the form (subject, predicate, object).
    Example:
    Text: "The Eiffel Tower is located in Paris, France, and was completed in 1889."
    Output:
    1. (Eiffel Tower, is located in, Paris)
    2. (Paris, is in, France)
    3. (Eiffel Tower, was completed in, 1889)
    """
    
    # Template for formatting prompts
    prompt_template = """
    Generate Triples for the following text:
    {text}
    """
    
    # Optional model generation parameters
    model_generate_parameters = {
        "temperature": 0.1,
        "max_new_tokens": 512,
       
    }
    
    # Maximum number of chunks to process (optional)
    max_chunks = 1

    # Initialize and run the generator with all parameters
    generator = TripleGenerator(
        api_key="",  # Add your API key if needed
        input_dir=input_dir,
        output_dir=output_dir,
        system_message=system_message,
        prompt_template=prompt_template,
        model_name=model_name,
        max_chunks=max_chunks,
        model_generate_parameters=model_generate_parameters
    )
    generator.process()