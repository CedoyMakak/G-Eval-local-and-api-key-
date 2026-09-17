from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PORTS = (8001, 8000, 8002, 8010)


def main() -> None:
    os.chdir(ROOT)
    python = _ensure_venv()
    _ensure_env()
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    print()
    print("Запускаю сервис оценки ответов LLM")
    print(f"Окно: {url}")
    print("Остановка: закрой это окно или Ctrl+C")
    print()

    proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
    )
    try:
        if _wait_ready(url):
            webbrowser.open(url)
        else:
            print("Сервер не ответил за 20 секунд. Открой ссылку сам:", url)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def _ensure_venv() -> Path:
    if VENV_PY.exists():
        return VENV_PY
    print("Создаю виртуальное окружение .venv ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    print("Ставлю зависимости, подожди минуту ...")
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    return VENV_PY


def _ensure_env() -> None:
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env.exists() and example.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("Создан .env из .env.example — при необходимости впиши ключ OpenRouter.")


def _free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("Не нашёл свободный порт (8001/8000/8002/8010).")


def _wait_ready(url: str) -> bool:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.4)
    return False


if __name__ == "__main__":
    main()
