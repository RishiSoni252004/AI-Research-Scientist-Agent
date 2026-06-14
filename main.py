import os
from dotenv import load_dotenv
from ingestion.arxiv_fetcher import ArxivFetcher
from knowledge.vector_store import VectorStoreManager
from agents.gap_analyzer import GapAnalyzer
from agents.hypothesis_generator import HypothesisGenerator
from agents.experiment_designer import ExperimentDesigner
from execution.sandbox_runner import SandboxRunner
from agents.evaluator import Evaluator

load_dotenv()

def main():
    print("=== AI Research Scientist Agent Starting ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. The agents will not be able to run.")
        print("Please create an .env file with your OPENAI_API_KEY.")
        return

    # 1. Ingestion
    fetcher = ArxivFetcher()
    topic = "reinforcement learning from human feedback"
    print(f"\n[1] Fetching papers for topic: {topic}")
    papers = fetcher.search_and_download(topic, max_results=2)
    
    # 2. Knowledge Storage
    print("\n[2] Storing papers in Vector DB")
    vsm = VectorStoreManager()
    vsm.add_papers(papers)
    
    # 3. Gap Analysis
    print("\n[3] Identifying Research Gaps")
    gap_analyzer = GapAnalyzer(vsm)
    gaps = gap_analyzer.analyze(topic)
    print(f"--- Identified Gaps ---\n{gaps}\n")
    
    # 4. Hypothesis Generation
    print("\n[4] Generating Hypothesis")
    hypothesis_gen = HypothesisGenerator()
    hypothesis = hypothesis_gen.generate(gaps)
    print(f"--- Proposed Hypothesis ---\n{hypothesis}\n")
    
    # 5. Experiment Design
    print("\n[5] Designing Experiment (Generating Code)")
    designer = ExperimentDesigner()
    code = designer.design(hypothesis)
    print(f"--- Generated Experiment Code ---\n{code}\n")
    
    # 6. Execution
    print("\n[6] Executing Experiment in Sandbox")
    runner = SandboxRunner()
    results = runner.execute(code)
    print(f"--- Execution Results ---\nSuccess: {results['success']}")
    if not results['success']:
        print(f"Error: {results['stderr']}\n")
    else:
        print(f"Stdout: {results['stdout'][:500]}...\n")
    
    # 7. Evaluation
    print("\n[7] Evaluating Final Results")
    evaluator = Evaluator()
    evaluation = evaluator.evaluate(hypothesis, results)
    print(f"--- Final Evaluation ---\n{evaluation}\n")
    
    print("=== AI Research Scientist Agent Finished ===")

if __name__ == "__main__":
    main()
