import sys
import hydra
import logging
from omegaconf import DictConfig
from chatbot import ChatBot

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Launch the chatbot
        chatbot = ChatBot(cfg)
        chatbot.launch()
    except Exception as e:
        logging.error(f"{e}")
        sys.exit()

if __name__ == "__main__":
    main()