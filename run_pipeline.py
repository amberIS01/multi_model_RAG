import subprocess
import sys
import time
from typing import List, Tuple

def run_command(command: str, description: str) -> bool:
    print(f"\n{description}")
    print("-" * 40)
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error: {description} failed")
        return False

    print(f"Completed: {description}")
    return True

def main() -> None:
    start_time = time.time()
    print("=" * 50)
    print("Multi-Modal RAG Pipeline")
    print("=" * 50)

    steps: List[Tuple[str, str]] = [
        ("python config.py", "Creating directories"),
        ("python process_document.py", "Extracting document data"),
        ("python create_embeddings.py", "Creating embeddings"),
    ]

    total_steps = len(steps)
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[Step {i}/{total_steps}]")
        if not run_command(command, description):
            print(f"\nPipeline failed at step {i}/{total_steps}: {description}")
            print("Please check the error messages above and fix the issue.")
            sys.exit(1)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print("Run 'streamlit run app.py' to start the application")
    print("=" * 50)

if __name__ == "__main__":
    main()