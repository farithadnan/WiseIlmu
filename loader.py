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
        Get VectorStore (db) from list of documents.

        Args:
            chunk_size (int): The size of the chunks. 
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            Chroma VectorStore.
        '''
        try:
            
            # Load documents
            loader = DirectoryLoader(self.documents_dir)
            loaded_dir = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(documents=loaded_dir)

            # Convert to vector space
            embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
            db = Chroma.from_documents(documents, embeddings)
            return db
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory {self.documents_dir} not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error while loading & splitting the documents: {e}")


    