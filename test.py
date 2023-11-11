import hydra
import pprint
import chromadb
from halo import Halo
from loader import DocHandler
from omegaconf import DictConfig
from llm_openai import OpenAIHandler
from langchain.vectorstores.chroma import Chroma
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


def generate_response(db: Chroma, qa, messages):
    # Create a loading spinner
    spinner = Halo(text='Loading...', spinner='dots')
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
    chroma_client = chromadb.Client()
    embedding_function = ONNXMiniLM_L6_V2()

    collection = chroma_client.create_collection(name="conversations", embedding_function=embedding_function)
    current_id = 0

    # Get vector database
    doc_handler = DocHandler(cfg)
    db = doc_handler.get_vector_db()

    # Initialize Q & A chain for OpenAI
    open_ai = OpenAIHandler(cfg, cfg.openAI.temperature, cfg.openAI.max_tokens)
    qa = open_ai.setup_chain()
    
    # Continue chatting until the user types "quit"
    while True:
        chat_history = []
        chat_metadata = []
        history_ids = []

        messages=[
        {"role": "system", "content": cfg.openAI.chat_persona}
        ]

        input_text = input("You: ")
        if input_text.lower() == "quit":
            break

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
        response = generate_response(db, qa, messages)

        # Update chat history & metadata
        chat_metadata.extend([{"role":"user"}, {"role": "assistant"}])
        chat_history.extend([input_text, response])

        # Update history IDs
        current_id += 1
        history_ids.extend([f"id_{current_id}", f"id{current_id + 1}"])

        # Add the conversation to the collection
        collection.add(
            documents=chat_history,
            metadatas=chat_metadata,
            ids=history_ids
        )

        # Print the assistant's response
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()