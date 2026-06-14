import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Evaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3) if os.getenv("OPENAI_API_KEY") else None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI peer reviewer. You are given the original hypothesis and the stdout/stderr execution logs from the experiment designed to test it. Evaluate whether the experiment succeeded, what the results mean, and whether the hypothesis is supported by the data."),
            ("user", "Hypothesis:\n{hypothesis}\n\nExecution Success: {success}\n\nStandard Output:\n{stdout}\n\nStandard Error:\n{stderr}\n\nPlease provide a comprehensive evaluation report.")
        ])
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def evaluate(self, hypothesis, execution_results):
        if not self.llm:
            return "Error: OPENAI_API_KEY is not set."
            
        print("Evaluating experiment results...")
        return self.chain.invoke({
            "hypothesis": hypothesis,
            "success": execution_results["success"],
            "stdout": execution_results["stdout"][:2000],  # truncate if too long
            "stderr": execution_results["stderr"][:2000]
        })

if __name__ == "__main__":
    print("Evaluator ready.")
