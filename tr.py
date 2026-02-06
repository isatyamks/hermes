from supervisor.ollama_client import OllamaClient
from supervisor.analyzer import OfflineAnalyzer
from supervisor.save_analysis import save_analysis
import os

# llm = OllamaClient(model="llama3:8b")
llm = OllamaClient(model="llama3.1:8b")

analyzer = OfflineAnalyzer(llm)

report_path = "reports/run_2025-12-19_15-15-00.json"

print("="*80)
print("Running LLM Analysis")
print("="*80)

analysis = analyzer.analyze(report_path)

analysis_path = save_analysis(analysis)
print(f"Analysis saved to: {os.path.abspath(analysis_path)}")

 

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)


