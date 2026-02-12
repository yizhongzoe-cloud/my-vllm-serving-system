"""Dataset loading and management for SLO benchmarks."""

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Request:
    """A single benchmark request with SLO requirements."""

    request_id: str
    prompt: str
    ttft_slo_ms: float | None = None
    e2e_latency_slo_ms: float | None = None
    max_tokens: int = 100

    # Optional metadata
    category: str | None = None
    priority: int = 0

    def to_api_payload(self, model: str) -> dict:
        """Convert to vLLM OpenAI API request format."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.ttft_slo_ms is not None:
            payload["ttft_slo_ms"] = self.ttft_slo_ms
        if self.e2e_latency_slo_ms is not None:
            payload["e2e_latency_slo_ms"] = self.e2e_latency_slo_ms
        return payload


class Dataset:
    """Dataset container for benchmark requests."""

    def __init__(self, requests: list[Request], name: str = "unnamed"):
        self.requests = requests
        self.name = name

    def __len__(self) -> int:
        return len(self.requests)

    def __iter__(self):
        return iter(self.requests)

    def __getitem__(self, idx: int) -> Request:
        return self.requests[idx]

    def shuffle(self, seed: int | None = None) -> "Dataset":
        """Return a shuffled copy of the dataset."""
        requests = self.requests.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(requests)
        return Dataset(requests, self.name)

    def subset(self, n: int) -> "Dataset":
        """Return first n requests."""
        return Dataset(self.requests[:n], self.name)

    def filter_by_slo(self, has_slo: bool = True) -> "Dataset":
        """Filter requests by whether they have SLO requirements."""
        filtered = [
            r for r in self.requests
            if (r.ttft_slo_ms is not None or r.e2e_latency_slo_ms is not None) == has_slo
        ]
        return Dataset(filtered, self.name)


class DatasetLoader:
    """Load datasets from various formats."""

    @staticmethod
    def from_jsonl(path: str | Path) -> Dataset:
        """
        Load dataset from JSONL file.

        Expected format per line:
        {
            "prompt": "...",
            "ttft_slo_ms": 100,        # optional
            "e2e_latency_slo_ms": 500, # optional
            "max_tokens": 100,         # optional, default 100
            "category": "...",         # optional
            "priority": 0              # optional
        }
        """
        path = Path(path)
        requests = []

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                req = Request(
                    request_id=data.get("request_id", f"req_{idx:06d}"),
                    prompt=data["prompt"],
                    ttft_slo_ms=data.get("ttft_slo_ms"),
                    e2e_latency_slo_ms=data.get("e2e_latency_slo_ms"),
                    max_tokens=data.get("max_tokens", 100),
                    category=data.get("category"),
                    priority=data.get("priority", 0),
                )
                requests.append(req)

        return Dataset(requests, name=path.stem)

    @staticmethod
    def from_json(path: str | Path) -> Dataset:
        """Load dataset from JSON file (array of requests)."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        requests = []
        for idx, data in enumerate(data_list):
            req = Request(
                request_id=data.get("request_id", f"req_{idx:06d}"),
                prompt=data["prompt"],
                ttft_slo_ms=data.get("ttft_slo_ms"),
                e2e_latency_slo_ms=data.get("e2e_latency_slo_ms"),
                max_tokens=data.get("max_tokens", 100),
                category=data.get("category"),
                priority=data.get("priority", 0),
            )
            requests.append(req)

        return Dataset(requests, name=path.stem)

    @staticmethod
    def generate_synthetic(
        n: int,
        slo_distribution: dict | None = None,
        prompt_templates: list[str] | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """
        Generate synthetic dataset for testing.

        Args:
            n: Number of requests to generate
            slo_distribution: Dict with keys:
                - "ttft_range": (min_ms, max_ms) or None
                - "e2e_range": (min_ms, max_ms) or None
                - "no_slo_ratio": fraction of requests without SLO
            prompt_templates: List of prompt templates to use
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        slo_distribution = slo_distribution or {
            "ttft_range": (50, 500),
            "e2e_range": (200, 3000),
            "no_slo_ratio": 0.2,
        }

        prompt_templates = prompt_templates or [
            "What is {topic}?",
            "Explain {topic} in simple terms.",
            "Write a short paragraph about {topic}.",
            "List 3 facts about {topic}.",
            "How does {topic} work?",
        ]

        topics = [
            "machine learning", "quantum computing", "climate change",
            "artificial intelligence", "blockchain", "renewable energy",
            "space exploration", "genetic engineering", "cybersecurity",
            "neural networks", "data science", "cloud computing",
        ]

        requests = []
        for i in range(n):
            template = random.choice(prompt_templates)
            topic = random.choice(topics)
            prompt = template.format(topic=topic)

            # Determine SLO
            ttft_slo = None
            e2e_slo = None

            if random.random() > slo_distribution.get("no_slo_ratio", 0.2):
                ttft_range = slo_distribution.get("ttft_range")
                e2e_range = slo_distribution.get("e2e_range")

                if ttft_range:
                    ttft_slo = random.uniform(*ttft_range)
                if e2e_range:
                    e2e_slo = random.uniform(*e2e_range)

            req = Request(
                request_id=f"syn_{i:06d}",
                prompt=prompt,
                ttft_slo_ms=ttft_slo,
                e2e_latency_slo_ms=e2e_slo,
                max_tokens=random.randint(50, 200),
            )
            requests.append(req)

        return Dataset(requests, name="synthetic")
