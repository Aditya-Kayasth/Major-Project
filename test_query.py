import time
import torch

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.45, 0)

from src.rag_pipeline import IndianLegalRAGPipeline

pipe = IndianLegalRAGPipeline()
query = (
    "Can a Special Court under POCSO issue bailable warrants or compel "
    "the physical attendance of a minor victim during bail hearings of the accused?"
)

print("\n--- RUNNING LEGAL FUNNEL ON RTX 2050 ---")

# Resilient query execution with automatic retry on 503 / 429
max_retries = 3
result = None

for attempt in range(1, max_retries + 1):
    result = pipe.run(query)
    answer = result.get("answer", "")
    if "503 UNAVAILABLE" not in answer and "429" not in answer:
        break
    print(f"[!] Upstream API busy (Attempt {attempt}/{max_retries}). Retrying in 5 seconds...")
    time.sleep(5)

print("\n" + "=" * 80)
print("FINAL SYNTHESIZED JUDICIAL OPINION:")
print("=" * 80)
print(result.get("answer", ""))

print("\n" + "=" * 80)
print("PIPELINE LATENCY BREAKDOWN:")
print("=" * 80)
for stage, sec in result.get("timings", {}).items():
    if isinstance(sec, (int, float)):
        print(f"  * {stage:<25}: {sec * 1000:>8.2f} ms ({sec:.2f} s)")

print("\n" + "=" * 80)
print("RETRIEVED PARENT CHUNKS:")
print("=" * 80)
for i, parent in enumerate(result.get("retrieved_parents", [])[:2], 1):
    text = parent.get("text", str(parent)) if isinstance(parent, dict) else str(parent)
    print(f"[Parent {i}]:\n{text[:280].strip()}...\n")