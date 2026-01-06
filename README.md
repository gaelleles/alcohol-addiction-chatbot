# Alcohol Addiction Chatbot

RAG+Streamlit chatbot specialised on addiction and alcohol issues.
This serves as a helper for a research project on addiction and alcohol-related issues.

# Setup

## Package manager

We use uv as the package and project manager for its speed. If you need to install it, refer to their [installation guide](https://docs.astral.sh/uv/getting-started/installation/). 

## .env and API key

Run `cp .env.example .env` and fill the necessary variables in `.env`.

For the API key : Retrieve an API key from [GROQ](https://console.groq.com/) (the free tier is quite generous). 

## Pre-commit

We use [pre-commit](https://pre-commit.com/#install) to ensure code quality. 

## IP

To protect the authors' intellectual property, I chose not to add the full list of papers to the public project. However, I left a sample of public papers that you can use to test the code. This is the "PRIVACY" setting in `.env`. Feel free to change it to `private` if you build your own library of papers in `pdf_papers/private`.

I encourage you to build a library that is mindful of the intellectual property of authors.

## Run the project

`uv run streamlit run app.py`

# Tech challenges

"J'ai d'abord utilisé FAISS, mais j'ai rencontré un problème de segmentation/indexation avec l'interpréteur Python 3.13 et NumPy 2.0. J'ai donc dû debugger la couche d'abstraction de LangChain pour implémenter soit une injection manuelle via from_texts, soit basculer sur ChromaDB pour assurer la stabilité du système."
