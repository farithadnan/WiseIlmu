import pprint
from halo import Halo
from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


class LLMOpenAI:
    def __init__(self):
        pass

    def get_llm(self, cfg: DictConfig):
        '''
        Method to get the LLM model.

        Args: 
            cfg (DictConfig): The configuration object.

        Returns:
            The LLM model.
        '''
        try:
            llm = ChatOpenAI(openai_api_key=cfg.openAI.api_key, model_name=cfg.openAI.model, 
                   temperature=cfg.openAI.temperature, max_tokens=cfg.openAI.max_tokens)
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
        
        return llm
    
    def get_qa_chain(self, cfg: DictConfig):
        '''
        Method to get the Q & A chain.

        Args:
            cfg (DictConfig): The configuration object.

        Returns:
            The Q & A chain.
        '''
        llm = self.get_llm(cfg)
        return load_qa_chain(llm=llm)
    
    def generate_response(self, vector_db: Chroma, qa_chain: BaseCombineDocumentsChain, messages):
        # Create a loading spinner
        spinner = Halo(text='Loading...', spinner='dots')
        spinner.start()

        # Fetch latest user Input
        latest_input = next((message for message in reversed(messages) if message.get("role") == "user"), None)

        # Get matching documents based on input text
        matching_docs = vector_db.similarity_search(latest_input["content"])
        answer = qa_chain.run(input_documents=matching_docs, question=messages)

        # Stop the spinner once the response is received
        spinner.stop()

        # Pretty-print the user message sent to the AI
        pp = pprint.PrettyPrinter(indent=4)
        print("Request:")
        pp.pprint(messages)

        return answer

