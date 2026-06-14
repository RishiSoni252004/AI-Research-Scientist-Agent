import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class GapAnalyzer:
    def __init__(self, vector_store_manager):
        self.vsm = vector_store_manager
        # We assume OPENAI_API_KEY is set in environment
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5) if os.getenv("OPENAI_API_KEY") else None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI research scientist. Your task is to analyze excerpts from recent research papers and identify unexplored research gaps, contradictions, or limitations that could form the basis of a novel research project."),
            ("user", "Here are excerpts from recent literature on a topic:\n\n{context}\n\nPlease identify 3 to 5 distinct research gaps. For each gap, provide a brief explanation of why it is significant and how it could be addressed.")
        ])
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def analyze(self, topic, collection_name="research_papers"):
        if not self.llm:
            return "Error: OPENAI_API_KEY is not set. Cannot run LLM."
            
        vector_store = self.vsm.get_vector_store(collection_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        docs = retriever.invoke(topic)
        context = "\n\n".join([f"Source: {doc.metadata.get('title', 'Unknown')}\nExcerpt: {doc.page_content}" for doc in docs])
        
        print(f"Analyzing {len(docs)} document chunks for topic: {topic}...")
        result = self.chain.invoke({"context": context})
        return result

if __name__ == "__main__":
    print("GapAnalyzer ready.")
