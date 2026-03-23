"""Server management utilities."""

import subprocess
import sys
import time
from pathlib import Path

import requests


class VLLMServer:
    """Manage vLLM server lifecycle."""

    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 8000,
        vllm_path: str | Path | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.vllm_path = Path(vllm_path) if vllm_path else self._find_vllm_path()
        self.extra_args = extra_args or []
        self.process: subprocess.Popen | None = None

    def _find_vllm_path(self) -> Path:
        """Find vLLM repo root (this repo itself is a vllm fork)."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "vllm" / "__init__.py").exists() and (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("Could not find vLLM repo root")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, wait: bool = True, timeout: int = 120) -> None:
        """Start the vLLM server."""
        if self.process is not None:
            raise RuntimeError("Server is already running")

        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--trust-remote-code",
            *self.extra_args,
        ]

        print(f"Starting vLLM server...")
        print(f"  Model: {self.model}")
        print(f"  URL: {self.base_url}")

        # execute cmd
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.vllm_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if wait:
            self.wait_ready(timeout=timeout)

    def wait_ready(self, timeout: int = 120) -> bool:
        """Wait for server to be ready."""
        print(f"Waiting for server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            if self.process and self.process.poll() is not None:
                stdout = self.process.stdout.read() if self.process.stdout else ""
                raise RuntimeError(f"Server process died. Output:\n{stdout}")

            time.sleep(1)
            print(".", end="", flush=True)

        print("\nServer startup timeout!")
        return False

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.process is None:
            return

        print("Stopping server...")
        self.process.terminate()

        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.process = None
        print("Server stopped.")

    def is_ready(self) -> bool:
        """Check if server is ready."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def check_server(host: str = "localhost", port: int = 8000) -> bool:
    """Check if a vLLM server is running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False
