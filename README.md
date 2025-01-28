# Project Overview
A robust system for automatically constructing knowledge graphs from unstructured text using advanced NLP techniques and Wikidata integration.

The project was part of the **Practical Course: Data Engineering**.

## Overview 📝

This project implements an end-to-end pipeline for:
1. Extracting content from web pages
2. Generating semantic triples using language models
3. Integrating generated triples with Wikidata ontology
4. Creating a queryable knowledge graph

## Performance Highlights 📊

Our system outperforms the standart Wikidata API in property matching :
![Property Matching Performance](images/Comparison of the Methods.png)

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
cd ./Knowledge-Graph-Creation-from-Text-with-LLMs
```

2. Install the dependencies
```bash
pip install -r requirements.txt
```

3. Create a config file / Modify the main config file

4. Run the pipeline
```bash
python run.py --config config/YourConfig.yaml --mode full
```

