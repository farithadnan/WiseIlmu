import sys
import hydra
import pprint
from halo import Halo

from omegaconf import DictConfig


from langchain.vectorstores.chroma import Chroma


from langchain.chains.combine_documents.base import BaseCombineDocumentsChain

from llm_openai import LLMOpenAI
from loader import Loader


def generate_response(db: Chroma, qa: BaseCombineDocumentsChain, messages):
    # Create a loading spinner
    spinner = Halo(text='Loading...\n', spinner='dots')
    spinner.start()

    # Fetch latest user Input
    latest_input = next((message for message in reversed(messages) if message.get("role") == "user"), None)

    # Get matching documents based on input text
    matching_docs = db.similarity_search(latest_input["content"])
    answer = qa.run(input_documents=matching_docs, question=messages)

    # Stop the spinner once the response is received
    spinner.stop()

    # Pretty-print the user message sent to the AI
    pp = pprint.PrettyPrinter(indent=4)
    print("Request:")
    pp.pprint(messages)

    return answer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Load documents
    documents_handler = Loader()
    documents = documents_handler.load_documents(cfg.documents_dir)

    # Split documents into chunks
    documents = documents_handler.split_documents(documents)

    # Create Chroma DB
    vector_db = documents_handler.create_vector_db(documents, cfg)

    # Create Chain
    openai_handler = LLMOpenAI()
    qa_chain = openai_handler.get_qa_chain(cfg)

    # Load/Create collection 
    collection = documents_handler.load_collection(cfg.vector_db_dir)

    current_id = 0
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the DocBot. You are now ready to start interacting with your documents')
    print('---------------------------------------------------------------------------------')

    while True:
        chat_history = []
        chat_metadata = []
        history_ids = []

        messages = [{"role": "system", "content": cfg.openAI.chat_persona}]

        input_text = input(f"{green}You: ")

        if input_text == "exit" or input_text == "quit":
            print('Exiting')
            sys.exit()
        if input_text == '':
            continue


        results = collection.query(
            query_texts=[input_text],
            where={"role": "assistant"},
            n_results=2
        )

        # append the query result into the messages
        for res in results['documents'][0]:
            messages.append({"role": "user", "content": f"previous chat: {res}"})

        # Add the user's input to the messages
        messages.append({"role": "user", "content": input_text})
        response = generate_response(vector_db, qa_chain, messages)

        # Update chat history & metadata
        chat_metadata.extend([{"role":"user"}, {"role": "assistant"}])
        chat_history.extend([input_text, response])

        # Update history IDs
        current_id += 1
        history_ids.extend([f"id_{current_id}", f"id_{current_id + 1}"])

        # Add the conversation to the collection
        collection.add(
            documents=chat_history,
            metadatas=chat_metadata,
            ids=history_ids
        )

        # Print the bot's response
        print(f"{white}Bot: {response}")

if __name__ == "__main__":
    main()