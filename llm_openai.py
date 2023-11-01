from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.chroma import Chroma


class OpenAIHandler:
    def __init__(self, cfg: DictConfig, query: str, db: Chroma, 
                 chain_type="stuff", verbose="true"):
        self.api_key = cfg.openAI.api_key
        self.model = cfg.openAI.model
        self.temperature = cfg.openAI.temperature
        self.max_tokens = cfg.openAI.max_tokens
        self.query = query
        self.db = db
        self.chain_type = chain_type
        self.verbose = verbose

    def load_model(self):
        '''
        Loads the OpenAI model.
        '''
        return ChatOpenAI(openai_api_key=self.api_key, model_name=self.model, 
                          temperature=self.temperature, max_tokens=self.max_tokens)
    

    def setup_chain(self):
        '''
        Sets up the Q & A chain.
        '''
        llm = self.load_model()
        return load_qa_chain(llm=llm, chain_type=self.chain_type, verbose=self.verbose)
    

    def get_response(self):
        '''
        Method to generate response from the query.
        '''
        matching_docs = self.db.similarity_search(self.query)
        chain = self.setup_chain()
        answer = chain.run(input_documents=matching_docs, question=self.query)
        return answer
