# FindIt AI

AI-Powered Fashion Visual Search Engine.

## Features

- Text-to-Image Search
- Image-to-Image Search
- CLIP-based semantic image embeddings
- FAISS similarity search
- Product filtering by gender, category, and color
- Similarity scores and ranked results

## Project Architecture

1. Fashion product images and metadata
2. Data cleaning and validation
3. CLIP image embedding extraction
4. FAISS vector indexing
5. Text and image retrieval
6. Streamlit user interface

## Requirements

- Python 3.10+
- Dependencies listed in requirements.txt

## Installation

Create and activate a virtual environment, then run:

pip install -r requirements.txt

## Dataset Images

Download the Fashion Product Images (Small) dataset from Kaggle:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Extract the dataset and locate its images folder.

## Run the Application

Run:

streamlit run app.py

Enter the full path to the dataset images folder in the application sidebar.

## Search Modes

- Text Search: describe a fashion product using natural language.
- Image Search: upload an image to retrieve visually similar products.

## Model

openai/clip-vit-base-patch32

Embedding dimension: 512

## Search Engine

FAISS IndexIDMap2(IndexFlatIP)

Similarity metric: Cosine Similarity over L2-normalized embeddings.

## Evaluation

The current evaluation uses articleType as a proxy relevance label.
It should not be interpreted as human-judged semantic relevance.

## Important

The project ZIP includes the model, processor, metadata, and FAISS index.
The original dataset images are not included.
