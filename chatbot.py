import gradio as gr
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler

class ChatBot: 
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.documents_dir = cfg.documents_dir
        self.embedding_model = cfg.embeddings.model

    def setup(self):
        '''
        Sets up the chatbot.
        
        Returns:
            The chatbot.
        '''
        try:
            # Get vector database
            docHandler = DocHandler(self.cfg)
            db = docHandler.get_vector_db()

            def chatbot(input_text):
                # Initialize Q & A chain for OpenAI
                openAI = OpenAIHandler(self.cfg)
                qa = openAI.setup_chain()

                # Get matching documents based on input text
                matching_docs = db.similarity_search(input_text)
                answer = qa.run(input_documents=matching_docs, question=input_text)
                return answer

            return chatbot
        except Exception as e:
            raise RuntimeError(f"Error while setting up the chatbot: {e}")

    def launch(self):
        '''
        Launches the chatbot.
        '''
        iface = gr.Interface(
            fn=self.setup(),
            inputs=gr.components.Textbox(lines=7, label="Question From User", placeholder="Enter your question here"),
            outputs=gr.components.Textbox(lines=7, label="Response from Chatbot", placeholder="Chatbot's response will appear here"),
            title="🤖Chatbot (OpenAI)"
        )
        iface.launch()




# Calculate using tiktoken