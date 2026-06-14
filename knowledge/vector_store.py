import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, persist_directory="data/chroma_db"):
        self.persist_directory = persist_directory
        
        # Use OpenAI Embeddings if key exists, else default to HuggingFace
        if os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings()
            print("Using OpenAI Embeddings.")
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Using HuggingFace Embeddings.")
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def add_papers(self, papers, collection_name="research_papers"):
        texts = []
        metadatas = []
        for paper in papers:
            if not paper.get('text'):
                continue
            chunks = self.text_splitter.split_text(paper['text'])
            texts.extend(chunks)
            for i in range(len(chunks)):
                metadatas.append({
                    "id": paper['id'],
                    "title": paper['title'],
                    "source": paper.get('pdf_url', ''),
                    "chunk_index": i
                })
        
        if not texts:
            print("No text found to add to vector store.")
            return None

        # Initialize or add to Chroma
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        # Note: Chroma >= 0.4 automatically persists if persist_directory is passed, 
        # but calling persist() is sometimes needed in older versions.
        if hasattr(vector_store, 'persist'):
            vector_store.persist()
            
        return vector_store
        
    def get_vector_store(self, collection_name="research_papers"):
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

if __name__ == "__main__":
    manager = VectorStoreManager()
    print("VectorStoreManager initialized successfully.")
