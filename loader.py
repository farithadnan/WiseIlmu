import chromadb
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

class DocHandler:
    def get_vector_db(self, embedding_model: str, document_dir: str, 
                      chunk_size=1000, chunk_overlap=20):
        '''
        Fetch vector representation of the documents in the directory.

        Args:
            embedding_model (str): The name of the embedding model.
            document_dir (str): The directory containing the documents.
            chunk_size (int): The size of the chunks.
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            Vector representation of the documents.
        '''
        documents = self.process_docs(document_dir, chunk_size, chunk_overlap)
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        db = Chroma.from_documents(documents, embeddings)
        return db


    def process_docs(self, document_dir: str, chunk_size=1000, chunk_overlap=20):
        '''
        Load & split the documents into smaller chunks.

        Args:
            chunk_size (int): The size of the chunks.
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            List of documents.
        '''
        # Load documents
        loader = DirectoryLoader(document_dir)
        documents = loader.load()
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents=documents)

        return docs
    