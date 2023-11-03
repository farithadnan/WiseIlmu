import gradio as gr
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler

class ChatBot: 
    def launch_chatbot(self, chatbot):
        try:
            iface = gr.Interface(
                fn=chatbot,
                inputs=gr.components.Textbox(lines=7, label="Question From User", placeholder="Enter your question here"),
                outputs=gr.components.Textbox(lines=7, label="Response from Chatbot", placeholder="Chatbot's response will appear here"),
                title="🤖Chatbot (OpenAI)"
            )
            iface.launch()
        except Exception as e:
            print(f"Error while launching the chatbot: {e}")

    def setup_chatbot(self, cfg: DictConfig, document_dir: str, embedding_model: str):
        '''
        Sets up the chatbot.

        Args:
            cfg (DictConfig): The configuration.
            document_dir (str): The directory containing the documents.
            embedding_model (str): The name of the embedding model.

        Returns:
            The chatbot.
        '''
        try:
            # Get vector database
            docHandler = DocHandler()
            db = docHandler.get_vector_db(embedding_model, document_dir)

            def chatbot(input_text):
                try:
                    # Initialize Q & A chain for OpenAI
                    openAI = OpenAIHandler()
                    qa = openAI.setup_chain(cfg)

                    # Get matching documents based on input text
                    matching_docs = db.similarity_search(input_text)
                    answer = qa.run(input_documents=matching_docs, question=input_text)
                    return answer
                except Exception as e:
                    raise RuntimeError(f"Error processing chatbot request: {e}")

            return chatbot
        except Exception as e:
            raise RuntimeError(f"Error while setting up the chatbot: {e}")



# Calculate using tiktoken