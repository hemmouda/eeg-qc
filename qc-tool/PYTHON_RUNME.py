import subprocess
import sys
import venv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
MAIN_SCRIPT = BASE_DIR / "src" / "main.py"
REQUIREMENTS_FILE = BASE_DIR / ".requirements.txt"


def check_tkinter():
    try:
        import tkinter
    except ImportError:
        print("=" * 60)
        print("ERROR: Your Python installation does not have tkinter.")
        print(f"Interpreter in use: {sys.executable}")
        print("Please see README.md for instructions on how to fix this.")
        print("=" * 60)
        sys.exit(1)


def check_requirements_file():
    if not REQUIREMENTS_FILE.exists():
        print("=" * 60)
        print(f"ERROR: {REQUIREMENTS_FILE.name} not found next to this script.")
        print("This file is required and must list the dependencies to install.")
        print("Please see README.md for more information.")
        print("=" * 60)
        sys.exit(1)


def get_venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv_and_install():
    print(f"Creating virtual environment using: {sys.executable}")
    venv.create(VENV_DIR, with_pip=True)

    venv_python = get_venv_python(VENV_DIR)

    print("Upgrading pip...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True
    )

    print(f"Installing dependencies from {REQUIREMENTS_FILE.name}...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        check=True,
    )


def main():
    check_tkinter()
    check_requirements_file()

    venv_exists = VENV_DIR.exists()

    if not venv_exists:
        create_venv_and_install()

    venv_python = get_venv_python(VENV_DIR)

    if not venv_python.exists():
        print(f"Error: venv python not found at {venv_python}")
        sys.exit(1)

    if not MAIN_SCRIPT.exists():
        print(f"Error: {MAIN_SCRIPT} not found")
        sys.exit(1)

    result = subprocess.run([str(venv_python), str(MAIN_SCRIPT)])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
