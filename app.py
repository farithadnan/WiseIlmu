import os
import hydra
from llm_openai import OpenAIHandler
from omegaconf import DictConfig
from loader import DocHandler

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Get directory path of the documents
    current_dir = os.path.dirname(os.path.abspath(__file__))
    document_dir = os.path.join(current_dir, "documents")

    # Get vector database
    docHandler = DocHandler(document_dir)
    db = docHandler.get_vector_db()

    query = "THIS IS YOUR QUERY"

    openAI = OpenAIHandler(cfg=cfg, query=query, db=db)
    answer = openAI.get_response()
    answer

if __name__ == "__main__":
    main()