import tiktoken
import gradio as gr
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler
from halo import Halo

class ChatBot: 
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.temperature = cfg.openAI.temperature
        self.max_tokens = cfg.openAI.max_tokens
        self.chat_cost_per_1000_tokens = cfg.openAI.chat_cost_per_1000_tokens
        self.bot_persona = cfg.openAI.chatbot_persona

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

            def chatbot(input_text, temperature, max_tokens):
                # Creat a loading spinner
                spinner = Halo(text="Loading...", spinner="dots")
                spinner.start()

                # Initialize Q & A chain for OpenAI
                openAI = OpenAIHandler(self.cfg, temperature, max_tokens)
                qa = openAI.setup_chain() # add message here

                # Get matching documents based on input text
                matching_docs = db.similarity_search(input_text)
                answer = qa.run(input_documents=matching_docs, question=input_text)

                # Fetch total word count, total token count, and estimated cost
                total_word_count, total_token_count, estimated_cost = self.check_tokens(input_text, answer)

                # Stop the loading spinner
                spinner.stop()
                return answer, total_word_count, total_token_count, estimated_cost

            return chatbot
        except Exception as e:
            gr.Error(f"Error while setting up the chatbot: {e}")
            raise RuntimeError(f"Error while setting up the chatbot: {e}")

    def launch(self):
        '''
        Launches the chatbot.
        '''
        # Input fields
        input_field = gr.components.Textbox(lines=7, label="🗣️ Question From User", placeholder="Enter your question here")
        temp_field = gr.components.Slider(minimum=0, maximum=2, step=0.1, label="Temperature", show_label=True, value=self.temperature)
        max_tokens_field = gr.components.Slider(minimum=1, maximum=4096, step=1, label="Maximum Output Length", show_label=True, value=self.max_tokens)

        # Output fields
        answer_field = gr.components.Textbox(lines=7, label="🧠 Response From Chatbot", placeholder="Chatbot's response will appear here")
        total_words_field = gr.components.Number(label="Total Words", show_label=True)
        total_tokens_field = gr.components.Number(label="Total Tokens", show_label=True)
        estimated_cost_field = gr.components.Number(label="Estimated Cost ($)", show_label=True)

        with gr.Blocks(title="AI Chatbot") as iface:
            gr.Interface(
                fn=self.setup(),
                inputs=[input_field, temp_field, max_tokens_field],
                outputs=[answer_field, total_words_field, total_tokens_field, estimated_cost_field],
                title="🤖 OpenAI Powered Knowledge Base"
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