# WiseIlmu 🧠

WiseIlmu is a Python application that leverages large language model (LLM) like OpenAI and the langchain library to load and split documents into chunks. WiseIlmu also uses the [sentence-transformers model (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to convert document chunks into a vector space format, which helps LLMs to better understand the content of documents. The vector space of the documents is then stored in memory by [Chroma](https://docs.trychroma.com/) allowing its utilization throughout the application's runtime for seamless querying.

## How to Run the Project 🏃🏽‍♂️

**Step 1: Clone the repo**

Open Git bash and type:
```
  git clone https://github.com/farithadnan/WiseIlmu.git
```

**Step 2: Installation** 

Install the required Python packages by running below command on your terminal:
  ```bash
    pip install -r requirements.txt
  ```

**Step 3: Configuration Setup**

Ensure that the configuration file (e.g., config.yaml) contains essential settings such as:
- Paths to directories containing documents in `app.py`
```python
  document_dir = os.path.join(current_dir, "documents")
```
- LLM Model and embedding details.

**Step 4: Run the Project**

Activate your Python environment and execute the main Python script:
```bash
  python app.py
```

This will initialize the chatbot using the configured settings and launch a Gradio-based interface for interacting with the OpenAI-based chatbot.