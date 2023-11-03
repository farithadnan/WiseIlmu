import gradio as gr
import tiktoken
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler

class ChatBot: 
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.chat_cost_per_1000_tokens = cfg.openAI.chat_cost_per_1000_tokens

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

                # Fetch total word count, total token count, and estimated cost
                total_word_count, total_token_count, estimated_cost = self.check_tokens(input_text, answer)
                return answer, total_word_count, total_token_count, estimated_cost

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
            outputs=[gr.components.Textbox(label="Answer"), gr.components.Textbox(label="Total Words"), 
                     gr.components.Textbox(label="Total Tokens"), gr.components.Textbox(label="Estimated Cost")],
            title="🤖Chatbot (OpenAI)"
        )
        iface.launch()

    def check_tokens(self, prompt: str, completion: str):
        '''
        Method to check the number of tokens

        Args:
            prompt: The prompt text
            completion: The completion text
        
        Returns:
            The total number of words, the total number of tokens, and the estimated cost
        '''
        # Calculate total number of tokens
        combined_text = prompt + " " + completion
        encoder = tiktoken.encoding_for_model(self.cfg.openAI.model)
        total_token_count = len(encoder.encode(combined_text))
        
        # Calculate estimated cost for chat
        estimated_cost = "{:.10f}".format(total_token_count * self.chat_cost_per_1000_tokens/1000)
        total_word_count = len(combined_text.split())

        return total_word_count, total_token_count, estimated_cost