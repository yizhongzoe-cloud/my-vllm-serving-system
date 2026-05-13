[Transition] Architecturally: client, router as a separate CPU-only process, and one engine process per GPU.
Engines publish KV checkpoints to a host-side store with two tiers — an in-process pinned-memory pool that backs sub-millisecond same-engine resume, and a /dev/shm mirror that other engines can read for cross-engine restore. Router watches engine health via a shm status file and reroutes in-flight requests when an engine goes silent. The key design fact: the same checkpoint data backs both same-engine preempt-and-resume and cross-engine recovery.

Key points: ① router separate from engines ② two-tier checkpoint store ③ one data path → two scheduling moves
Duration: 2 minutes