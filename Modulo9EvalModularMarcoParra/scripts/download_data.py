from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "heart.csv"

def have_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None

def main() -> int:
    if OUT.exists():
        print(f"✅ Ya existe: {OUT.resolve()}")
        return 0

    print("Intentando descargar con Kaggle CLI: fedesoriano/heart-failure-prediction ...")
    if not have_kaggle_cli():
        sys.stderr.write(
            "[!] No se encontró la herramienta 'kaggle'.\n"
            "    1) Instala la CLI:  pip install kaggle\n"
            "    2) Coloca tu API key en ~/.kaggle/kaggle.json (permisos 600)\n"
            "    3) Luego corre de nuevo:  python -m scripts.download_data\n"
            "    O copia manualmente el archivo heart.csv a data/raw/heart.csv\n"
        )
        return 1

    try:
        # Descargar y descomprimir en data/raw/
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "fedesoriano/heart-failure-prediction",
                "-p", str(RAW_DIR),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            "[!] Falló la descarga con Kaggle CLI.\n"
            "    Verifica que tengas credenciales válidas en ~/.kaggle/kaggle.json\n"
            "    y que aceptaste los términos del dataset en Kaggle.\n"
        )
        return e.returncode

    # Verificación final
    if OUT.exists():
        print(f"✅ Dataset descargado en: {OUT.resolve()}")
        return 0
    else:
        sys.stderr.write(
            "[!] No se encontró data/raw/heart.csv tras la descarga.\n"
            "    Revisa el contenido de data/raw/ y, si el archivo tiene otro nombre,\n"
            "    renómbralo a 'heart.csv'. Alternativamente, descarga manualmente desde Kaggle\n"
            "    y colócalo en data/raw/heart.csv.\n"
        )
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
