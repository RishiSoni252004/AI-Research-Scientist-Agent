import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ExperimentDesigner:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2) if os.getenv("OPENAI_API_KEY") else None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI software engineer and researcher. Your task is to write a complete, standalone Python script that executes an experiment to test the provided hypothesis. Use standard libraries (numpy, scikit-learn, etc.) and mock data if real datasets are too large to download. The script must print the final evaluation metrics clearly to stdout. Output ONLY valid Python code, starting with ```python and ending with ```."),
            ("user", "Hypothesis Details:\n{hypothesis}\n\nPlease write the Python script to run this experiment.")
        ])
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def design(self, hypothesis):
        if not self.llm:
            return "Error: OPENAI_API_KEY is not set."
            
        print("Designing experiment code...")
        output = self.chain.invoke({"hypothesis": hypothesis})
        
        # Extract code from markdown blocks
        code_match = re.search(r"```python\n(.*?)\n```", output, re.DOTALL)
        if code_match:
            return code_match.group(1)
        return output

if __name__ == "__main__":
    print("ExperimentDesigner ready.")
