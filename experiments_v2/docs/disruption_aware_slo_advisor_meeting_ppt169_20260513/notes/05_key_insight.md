[Transition] The whole design collapses to one sentence.
A single host-side KV checkpoint, async-published to /dev/shm at block-aligned cadence, simultaneously serves as the recovery substrate when an engine dies, and the cheap-preempt substrate that lets the scheduler preempt running long-context decode without wasting prefill. Same data, two uses, no duplication. The broader point — when a mechanism's cost drops below a threshold, design space that was previously off-limits opens up — is what we'd like to argue as the paper's framing contribution.

Key points: ① one checkpoint, two uses ② mechanism-policy coupling is the core design move ③ "cost threshold → new design space" is the framing story
Duration: 2 minutes