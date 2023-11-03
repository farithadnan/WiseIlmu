import hydra
import logging
from omegaconf import DictConfig
from chatbot import ChatBot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)

    try:
        # launch the chatbot
        chatbot = ChatBot(cfg)
        chatbot.launch()
    
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()