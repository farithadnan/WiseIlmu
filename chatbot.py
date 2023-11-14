import sys
import gradio as gr
from chromadb import Collection
from omegaconf import DictConfig
from langchain.vectorstores.chroma import Chroma
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


from llm_openai import LLMOpenAI

class ChatBot:
    def __init__(self, cfg: DictConfig, collection: Collection, 
                 vector_db: Chroma, qa_chain: BaseCombineDocumentsChain,):
        self.cfg = cfg
        self.collection = collection
        self.vector_db = vector_db
        self.qa_chain = qa_chain
    
    def launch(self):

        # Input fields
        temp_field = gr.components.Slider(minimum=0, maximum=2, step=0.1, label="Temperature", show_label=True, value=self.temperature)
        max_tokens_field = gr.components.Slider(minimum=1, maximum=4096, step=1, label="Maximum Output Length", show_label=True, value=self.max_tokens)

        with gr.Blocks(title="AI Chatbot") as iface:
            gr.ChatInterface(
                fn=self.chat_engine(),
                title="🤖 OpenAI Powered Knowledge Base",
                additional_inputs= [temp_field, max_tokens_field],
            )
        
        iface.launch()

    def chat_engine(self):

        def chatbot(input_text, history):
            chat_history = []
            chat_metadata = []
            history_ids = []

            messages = [{"role": "system", "content": self.cfg.openAI.chat_persona}]

            if input_text == "exit" or input_text == "quit":
                print('Exiting...')
                sys.exit()
            if input_text == '':
                pass


            results = self.collection.query(
                query_texts=[input_text],
                where={"role": "assistant"},
                n_results=2
            )

            # append the query result into the messages
            for res in results['documents'][0]:
                messages.append({"role": "user", "content": f"previous chat: {res}"})

            # Add the user's input to the messages
            messages.append({"role": "user", "content": input_text})
            
            openai_handler = LLMOpenAI()
            response = openai_handler.generate_response(self.vector_db, self.qa_chain, messages)

            # Update chat history & metadata
            chat_metadata.extend([{"role":"user"}, {"role": "assistant"}])
            chat_history.extend([input_text, response])

            # Update history IDs
            current_id += 1
            history_ids.extend([f"id_{current_id}", f"id_{current_id + 1}"])

            # Add the conversation to the collection
            self.collection.add(
                documents=chat_history,
                metadatas=chat_metadata,
                ids=history_ids
            )
        
        return chatbot
