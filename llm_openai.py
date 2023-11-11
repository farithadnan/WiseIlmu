from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class OpenAIHandler:

    def __init__ (self, cfg: DictConfig, temperature, max_tokens):
        self.config = cfg.openAI
        self.temperature = temperature
        self.max_tokens = max_tokens

    def setup_chain(self, chain_type="stuff", verbose="false"):
        '''
        Sets up the Q & A chain.

        Args:
            chain_type (str): The type of the chain.
            verbose (str): Whether to print the logs or not.

        Returns:
            The Q & A chain.
        '''
        try:
            
            # Setup OpenAI chat
            llm = ChatOpenAI(openai_api_key=self.config.api_key, model_name=self.config.model, 
                                temperature=self.temperature, max_tokens=self.max_tokens) 
            
            # Setup Q & A Chain
            loaded_qa_chain = load_qa_chain(llm=llm, chain_type=chain_type, verbose=verbose)
            return loaded_qa_chain
    
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
