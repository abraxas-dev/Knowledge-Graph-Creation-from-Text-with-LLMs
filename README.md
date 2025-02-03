# Project Overview
A robust system for automatically constructing knowledge graphs from unstructured text using advanced NLP techniques and Wikidata integration.

The project was part of the **Practical Course: Data Engineering**.

**README IS NOT FINISHED YET**

## Overview 📝

This project implements an end-to-end pipeline for:
1. Extracting content from web pages
2. Generating semantic triples using language models
3. Integrating generated triples with Wikidata ontology
4. Creating a queryable knowledge graph

## Performance Highlights 📊

Our system outperforms the standart Wikidata API in property matching :
![Property Matching Performance](images/performance_for_methods.png)

## Developers :

**Didarbek Baidaliyev - abraxas-dev**
- System Architect
- Developed Integrator, Generator & Evaluator
- Data Analysis
- Documentation and testing

**Ar Pazari - paza15**
- Developed Extractor & Metrics
- Data Preparer

**Supervisor - (Samuel Garcia (https://github.com/Sondeluz))**
- Thanks to our supervisor for his guidance and support throughout the project :) 

## Key Features ✨

- **Flexible Content Extraction**: Configurable text chunking and preprocessing
- **Advanced Triple Generation**: Uses open source language models
- **Smart Property Matching**: 
  - Semantic similarity matching
  - Alias handling
  - Configurable matching strategies
- **Efficient Graph Management**: RDF-based storage and querying

## Usage 🚀
1. Clone the repository
```bash
git clone https://github.com/abraxas-dev/Knowledge-Graph-Creation-from-Text-with-LLMs.git
```

2. Navigate to the project directory
```bash
cd ./Knowledge-Graph-Creation-from-Text-with-LLMs
```

3. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

4. Install the dependencies
```bash
pip install -r requirements.txt
```

5. Modify the config file and run the pipeline
```bash
python run.py --config config/YourConfig.yaml --mode full
```

