from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

class DocHandler:

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def get_vector(self):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        documents = self.split_docs()
        db = Chroma.from_documents(documents, embeddings)
        return db

    def split_docs(self, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = self.load_docs()
        docs = text_splitter.split_documents(documents=documents)
        return docs
    
    def load_docs(self):
        loader = DirectoryLoader(self.directory_path)
        documents = loader.load()
        return documents
    



    

