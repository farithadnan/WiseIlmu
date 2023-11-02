from cycler import K
from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class OpenAIHandler:

    def load_model(self, cfg: DictConfig):
        '''
        Loads the OpenAI model.

        Returns:
            The OpenAI model.
        '''
        try:
            openAI = cfg.openAI
            return ChatOpenAI(openai_api_key=openAI.api_key, model_name=openAI.model, 
                                temperature=openAI.temperature, max_tokens=openAI.max_tokens)
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
    

    def setup_chain(self, cfg: DictConfig, chain_type="stuff", verbose="false"):
        '''
        Sets up the Q & A chain.

        Returns:
            The Q & A chain.
        '''
        try:
            llm = self.load_model(cfg)
            return load_qa_chain(llm=llm, chain_type=chain_type, verbose=verbose)
        except Exception as e:
            raise RuntimeError(f"Error while setting up the Q & A chain: {e}")
