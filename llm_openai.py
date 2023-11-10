from pyexpat import model
import re
import openai
from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class OpenAIHandler:

    def __init__ (self, cfg: DictConfig, temperature, max_tokens):
        self.cfg = cfg
        self.temperature = temperature
        self.max_tokens = max_tokens

    def setup_chain(self, messages: any, chain_type="stuff", verbose="false"):
        '''
        Sets up the Q & A chain.

        Args:
            chain_type (str): The type of the chain.
            verbose (str): Whether to print the logs or not.

        Returns:
            The Q & A chain.
        '''
        llm = self.load_model(messages)
        return load_qa_chain(llm=llm, chain_type=chain_type, verbose=verbose)

    def load_model(self, messages: any):
        '''
        Loads the OpenAI model.

        Returns:
            The OpenAI model.
        '''
        try:
            config = self.cfg.openAI
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            

            return response

            return ChatOpenAI(openai_api_key=config.api_key, model_name=config.model, 
                                temperature=self.temperature, max_tokens=self.max_tokens)
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
    