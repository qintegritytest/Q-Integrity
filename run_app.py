import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"

def python_in_venv():
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_venv():
    print("Creando entorno virtual 'venv'...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

def install_requirements(py_exec):
    req = ROOT / "requirements.txt"
    if req.exists():
        print("Instalando dependencias desde requirements.txt...")
        print("  Actualizando pip y instalando uv...")
        subprocess.check_call([str(py_exec), "-m", "pip", "install", "--upgrade", "pip", "uv"])
        print("  Usando uv para instalar paquetes (más rápido)...")
        subprocess.check_call([str(py_exec), "-m", "uv", "pip", "install", "-r", str(req)])
    else:
        print("No se encontró requirements.txt — omitiendo instalación.")

def run_streamlit(py_exec):
    print("Iniciando la app con Streamlit...")
    subprocess.check_call([str(py_exec), "-m", "streamlit", "run", str(ROOT / "app.py")])

def main():
    vpy = python_in_venv()

    try:
        if not VENV_DIR.exists() or not vpy.exists():
            create_venv()
            vpy = python_in_venv()
            install_requirements(vpy)
        run_streamlit(vpy)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando comando: {e}")
        sys.exit(e.returncode if hasattr(e, 'returncode') else 1)

if __name__ == '__main__':
    main()
