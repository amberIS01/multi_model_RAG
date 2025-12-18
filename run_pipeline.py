import subprocess
import sys
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
    print("=" * 50)
    print("Multi-Modal RAG Pipeline")
    print("=" * 50)

    steps: List[Tuple[str, str]] = [
        ("python config.py", "Creating directories"),
        ("python process_document.py", "Extracting document data"),
        ("python create_embeddings.py", "Creating embeddings"),
    ]

    for command, description in steps:
        if not run_command(command, description):
            print("\nPipeline failed. Please check the errors above.")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("Run 'streamlit run app.py' to start the application")
    print("=" * 50)

if __name__ == "__main__":
    main()