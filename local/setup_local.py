"""Setup script for local mode — checks dependencies and pulls Ollama model."""
import subprocess
import sys
import shutil


def check_python_deps():
    """Check and install Python dependencies for local mode."""
    deps = [
        "chromadb",
        "sentence-transformers",
        "tiktoken",
        "pypdf",
        "python-docx",
    ]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    else:
        print("All Python dependencies installed.")


def check_ollama():
    """Check if Ollama is installed and pull the model."""
    if not shutil.which("ollama"):
        print("\n⚠️  Ollama not found!")
        print("   Install it from: https://ollama.com/download")
        print("   Then run: ollama pull llama3.2")
        return False

    print("Ollama found. Pulling llama3.2 model (if not already downloaded)...")
    result = subprocess.run(["ollama", "pull", "llama3.2"], capture_output=False)
    if result.returncode != 0:
        print("⚠️  Failed to pull llama3.2. Make sure Ollama is running: 'ollama serve'")
        return False

    print("llama3.2 model ready.")
    return True


def main():
    print("=" * 50)
    print("  RAG Local Mode — Setup")
    print("=" * 50)

    print("\n1. Checking Python dependencies...")
    check_python_deps()

    print("\n2. Checking Ollama...")
    ollama_ok = check_ollama()

    print("\n" + "=" * 50)
    if ollama_ok:
        print("Setup complete! Run the local server:")
        print("  python local/local_server.py")
    else:
        print("Setup partially complete.")
        print("Install Ollama, then run: python local/local_server.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
