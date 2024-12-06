## Project Overview
This project focuses on automatically generating **Knowledge Graphs (KGs)** from **Wikipedia articles** using **Large Language Models (LLMs)**. The goal is to extract and structure knowledge from unstructured text, making it more accessible and usable for various applications.

The project is part of the **Practical Course: Data Engineering**.

## Project Structure
The project is divided into **four main components**, each addressing a key stage of the Knowledge Graph creation pipeline:

### 1. **Data Collection**
- Identify and extract relevant Wikipedia articles (e.g., "Soup").
- Preprocess and clean the text, removing unnecessary elements to ensure high-quality input data.

### 2. **Knowledge Extraction with LLMs**
- Leverage LLMs to:
  - Identify entities (e.g., people, places, concepts) from the text.
  - Extract relationships between these entities.
  - Summarize article sections to highlight core ideas.
- Tools: HuggingFace Transformers

### 3. **Knowledge Graph Construction**
- Convert the extracted entities and relationships into structured triples.
- Integrate these triples into a Knowledge Graph structure, using ontologies (e.g., DBpedia, Wikidata) to ensure semantic richness and consistency.
- Store the resulting graph in a suitable format or database.

### 4. **Evaluation and Visualization**
- Assess the coverage and accuracy of the generated Knowledge Graph.
- Perform queries to validate its utility and completeness.
- Visualize the graph to provide clear insights into the extracted knowledge and facilitate further analysis.

