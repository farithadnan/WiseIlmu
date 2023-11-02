import os
import hydra
import gradio as gr
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler

def launch_chatbot(chatbot):
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.components.Textbox(lines=7, label="Question From User", placeholder="Enter your question here"),
        outputs=gr.components.Textbox(lines=7, label="Response from Chatbot", placeholder="Chatbot's response will appear here"),
        title="🤖Chatbot (OpenAI)"
    )
    iface.launch()

def setup_chatbot(cfg: DictConfig, document_dir: str, embedding_model: str):
    '''
    Sets up the chatbot.

    Args:
        cfg (DictConfig): The configuration.
        document_dir (str): The directory containing the documents.
        embedding_model (str): The name of the embedding model.

    Returns:
        The chatbot.
    '''
    # Get vector database
    docHandler = DocHandler()
    db = docHandler.get_vector_db(embedding_model, document_dir)

    def chatbot(input_text):
        # Initialize Q & A chain for OpenAI
        openAI = OpenAIHandler()
        qa = openAI.setup_chain(cfg)

        # Get matching documents based on input text
        matching_docs = db.similarity_search(input_text)
        answer = qa.run(input_documents=matching_docs, question=input_text)
        return answer

    return chatbot


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Get directory path of the documents
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    document_dir = os.path.join(current_dir, "documents")
    chatbot = setup_chatbot(cfg, document_dir, cfg.embeddings.model)
    launch_chatbot(chatbot)

if __name__ == "__main__":
    main()