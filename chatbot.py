import gradio as gr
from omegaconf import DictConfig
from llm_openai import LLMOpenAI
from loader import Loader

class ChatBot:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.temperature = cfg.openAI.temperature
        self.max_tokens = cfg.openAI.max_tokens
    
    def launch(self):
        '''
        Method to launch the chatbot.
        '''
        # Input fields
        temp_field = gr.components.Slider(minimum=0, maximum=2, step=0.1, label="Temperature", show_label=True, value=self.temperature)
        max_tokens_field = gr.components.Slider(minimum=1, maximum=4096, step=1, label="Maximum Output Length", show_label=True, value=self.max_tokens)

        with gr.Blocks(title="AI Chatbot") as interface:
            gr.ChatInterface(
                fn=self.chat_engine(),
                title="🤖 OpenAI Powered Knowledge Base",
                additional_inputs= [temp_field, max_tokens_field],
            )
        
        interface.launch()

    def chat_engine(self):
        '''
        Method to handle the conversation between the user and the chatbot.

        Returns:
            A function that takes the user's input and returns the chatbot's response.
        '''

        # Load documents
        documents_handler = Loader()
        documents = documents_handler.load_documents(self.cfg.documents_dir)

        # Split documents into chunks
        documents = documents_handler.split_documents(documents)

        # Create Chroma DB
        vector_db = documents_handler.create_vector_db(documents, self.cfg)
        
        # Load/Create collection 
        collection = documents_handler.load_collection(self.cfg.vector_db_dir)

        def chatbot(input_text, history, temperature, max_tokens):
            chat_history = []
            chat_metadata = []
            history_ids = []
            current_id = 0

            # Create Chain
            openai_handler = LLMOpenAI(cfg=self.cfg, temperature=temperature, max_tokens=max_tokens)
            qa_chain = openai_handler.get_qa_chain()

            # Add persona to the bot
            messages = [{"role": "system", "content": self.cfg.openAI.chat_persona}]
            
            # Get the previous chat history
            results = collection.query(
                query_texts=[input_text],
                where={"role": "assistant"},
                n_results=2
            )

            # append the query result of previous chat into the messages
            for res in results['documents'][0]:
                messages.append({"role": "assistant", "content": f"previous chat: {res}"})

            # Get the current ID
            if len(results['ids'][0]) > 0:
                max_id_string = max(results['ids'][0], key=lambda x: int(x.split("_")[1]))
                max_id_number = int(max_id_string.split("_")[1])
                current_id = max_id_number

            # append log of user's input to the messages
            messages.append({"role": "user", "content": input_text})            
            response = openai_handler.generate_response(vector_db, qa_chain, messages)

            # Update chat history & metadata
            chat_metadata.extend([{"role":"user"}, {"role": "assistant"}])
            chat_history.extend([input_text, response])

            # Update history IDs
            current_id += 1

            # Check if the document with the same ID already exists
            existing_document = None
            for res in results['ids'][0]:
                if f"id_{current_id}" in res:
                    existing_document = res
                    break

            if not existing_document:
                history_ids.extend([f"id_{current_id}", f"id_{current_id + 1}"])
                collection.add(
                    documents=chat_history,
                    metadatas=chat_metadata,
                    ids=history_ids
                )

            return response
    
        return chatbot
