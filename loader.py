import chromadb
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

class DocHandler:
    def __init__ (self, cfg):
        self.documents_dir = cfg.documents_dir
        self.embedding_model = cfg.embeddings.model

    def get_vector_db(self, chunk_size=1000, chunk_overlap=20):
        '''
        Fetch vector representation of the documents in the directory.

        Args:
            chunk_size (int): The size of the chunks. 
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            Vector representation of the documents.
        '''
        documents = self.process_docs(chunk_size, chunk_overlap)
        embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        db = Chroma.from_documents(documents, embeddings)
        return db



    def process_docs(self, chunk_size=1000, chunk_overlap=20):
        '''
        Load & split the documents into smaller chunks.

        Args:
            chunk_size (int): The size of the chunks.
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            List of documents.
        '''
        try:
            # Load documents
            loader = DirectoryLoader(self.documents_dir)
            documents = loader.load()
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents=documents)

            return docs
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory {self.documents_dir} not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error while loading & splitting the documents: {e}")

    