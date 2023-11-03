from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class OpenAIHandler:

    def __init__ (self, cfg: DictConfig):
        self.cfg = cfg

    def load_model(self):
        '''
        Loads the OpenAI model.

        Returns:
            The OpenAI model.
        '''
        try:
            openAI = self.cfg.openAI
            return ChatOpenAI(openai_api_key=openAI.api_key, model_name=openAI.model, 
                                temperature=openAI.temperature, max_tokens=openAI.max_tokens)
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
    

    def setup_chain(self, chain_type="stuff", verbose="false"):
        '''
        Sets up the Q & A chain.

        Args:
            chain_type (str): The type of the chain.
            verbose (str): Whether to print the logs or not.

        Returns:
            The Q & A chain.
        '''
        llm = self.load_model()
        return load_qa_chain(llm=llm, chain_type=chain_type, verbose=verbose)

