import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class HypothesisGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.8) if os.getenv("OPENAI_API_KEY") else None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an innovative AI research scientist. Your goal is to propose a novel, concrete, and testable research hypothesis based on identified research gaps. The hypothesis should be something that can be evaluated programmatically through an automated experiment (e.g., training a model, evaluating an algorithm on a dataset)."),
            ("user", "Here are some identified research gaps in a specific area of AI:\n\n{gaps}\n\nPlease generate exactly ONE detailed research hypothesis. Include:\n1. The Hypothesis Statement\n2. The Rationale\n3. Proposed Methodology (high-level)\n4. Metrics for Evaluation")
        ])
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, gaps):
        if not self.llm:
            return "Error: OPENAI_API_KEY is not set."
            
        print("Generating hypothesis based on gaps...")
        return self.chain.invoke({"gaps": gaps})

if __name__ == "__main__":
    print("HypothesisGenerator ready.")
