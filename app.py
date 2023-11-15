import hydra
from omegaconf import DictConfig
from chatbot import ChatBot


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Launch the chatbot
    chatbot = ChatBot(cfg)
    chatbot.launch()

if __name__ == "__main__":
    main()