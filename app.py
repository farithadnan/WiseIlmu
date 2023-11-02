import os
import hydra
import logging
from omegaconf import DictConfig
from chatbot import ChatBot

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)

    try:
        # Get directory path of the documents
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        document_dir = os.path.join(current_dir, "documents")

        # Setup & launch the chatbot
        chatbot = ChatBot().setup_chatbot(cfg, document_dir, cfg.embeddings.model)
        chatbot.launch_chatbot(chatbot)
    
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()