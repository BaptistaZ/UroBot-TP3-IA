
# Installation Manual – ESTG_RegPedagogicoBot

## System Requirements

- Python 3.10+
- Miniconda (https://docs.conda.io/en/latest/miniconda.html)
- Ollama (https://ollama.com/)
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/BaptistaZ/UroBot-TP3-IA.git
cd UroBot-TP3-IA
```

## 2. Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate uroenv
```

## 3. Install and Run Ollama

1. Download and install Ollama for your operating system.
2. Run the following command:

```bash
ollama run llama3:8b
```

## 4. Prepare Resources

1. Place the file `ESTG_Regulamento2023.pdf` in the `resources/` folder.
2. Generate the vector database:

```bash
python llm_database.py
```

## 5. Verify Ollama Integration

Make sure the function `ollama_chat()` is implemented in the code to communicate with the local Ollama server.

## 6. Run the Flask Application

```bash
python UroBot_flask_app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## Conclusion

The ESTG_RegPedagogicoBot works entirely offline using:
- A local LLM via Ollama
- A Flask-based web interface
- A RAG system with ChromaDB
