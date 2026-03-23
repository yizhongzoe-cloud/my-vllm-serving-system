# Fault-Tolerant Multi-GPU LLM Serving with Adaptive KV-Cache Checkpointing

## 1. Core Idea

We consider a **single machine with multiple GPU replicas**, where each GPU loads the **same LLM model**. A scheduler routes incoming requests to one of these replicas for execution. During generation, each request periodically checkpoints its KV cache to **shared host memory (RAM)** so that its execution state can survive GPU failures.

When one or more GPUs become unavailable, the scheduler **re-routes affected requests** to surviving GPUs, restores the latest checkpointed KV state from host memory, and resumes generation there.

The key feature is that checkpointing is **adaptive** rather than fixed:

- when a request has just started, it may use **no checkpointing**;
- as generation progresses and more tokens are produced, its checkpoint level becomes stronger;
- equivalently, its checkpoint frequency increases over time.

We model this using checkpoint levels such as

- `0`: no checkpointing,
- `1`: low checkpoint frequency,
- `2`: high checkpoint frequency.

So the system jointly decides:

1. **which requests to admit**,
2. **where to place them initially**,
3. **how to reassign them after failures**,
4. **how aggressively to checkpoint their KV cache over time**.

---

## 2. Motivation

Modern LLM serving systems often rely on **replica scaling**: many identical GPU replicas are used to absorb high request volume and meet latency objectives. In this setting, user experience is mainly determined by two latency metrics:

- **TTFT (time-to-first-token)**: how quickly the first token appears;
- **TPOT (time-per-output-token)**: how smoothly the response continues streaming.

However, in practice, replicas may become unavailable due to events such as GPU reset, preemption, OOM-related instability, driver/runtime failures, or node-local issues. Once this happens, the system immediately loses usable serving capacity, which can cause:

- longer queues,
- TTFT and TPOT violations,
- interrupted token streams,
- reduced effective throughput or **goodput**.

A natural way to reduce recovery cost is to checkpoint request state. But checkpointing itself is not free:

- more frequent checkpointing increases memory traffic and runtime overhead;
- less frequent checkpointing reduces steady-state overhead but increases recovery delay and replay work after a failure.

This creates the central tradeoff of the problem:

> **How should the system jointly make routing, admission, and adaptive checkpointing decisions so that it remains feasible under failures while still maximizing useful serving throughput?**

---

## 3. Problem Setting

### System architecture

We study a **single-node, multi-GPU replicated serving system**:

- one machine contains a set of GPU replicas `r ∈ R`;
- all replicas load the **same model**;
- host memory is shared and is used to store checkpointed KV-cache state;
- a centralized scheduler dispatches requests and performs failover.

Each request `j` has:

- prompt length `P_j`,
- expected output length `G_j`,
- TTFT bound `D_j^{ttft}`,
- TPOT bound `D_j^{tpot}`,
- failure-gap bound `D_j^{gap}`.

Each replica `r` has:

- prefill throughput `C_r^{pre}`,
- decode throughput `C_r^{dec}`,
- recovery load bandwidth `B_r^{ld}`.

We use a planning horizon `H` for capacity accounting.

### Failure model

A failure scenario is denoted by `ω ⊆ R`, the set of failed GPUs. We consider a robust uncertainty set

```math
\Omega_k = \{\omega : |\omega| \le k\},
```

meaning the system must remain feasible under **any failure scenario with at most `k` failed GPUs**.

Replica availability under scenario `ω` is

```math
a_r(\omega) = \mathbf{1}[r \notin \omega].
```

All post-failure constraints must hold for every `ω ∈ Ω_k`.

---

## 4. Decision Variables

### Admission

```math
y_j \in \{0,1\}
```

indicates whether request `j` is admitted.

### Initial routing

```math
x_{j,r} \in \{0,1\}
```

indicates whether request `j` is initially assigned to replica `r`.

### Re-routing after failures

```math
\tilde{x}_{j,r}(\omega) \in \{0,1\}
```

indicates whether request `j` is assigned to replica `r` under failure scenario `ω`.

### Adaptive checkpoint level

A request does **not** need to use the same checkpoint level throughout its whole lifetime. Instead, checkpointing becomes stronger as the request grows longer.

We can describe this at two levels:

#### Simple abstract view

```math
\ell_j \in \{0,1,2\}
```

represents a checkpoint level associated with request `j`, where larger values mean stronger checkpointing.

#### More faithful dynamic view

Since your real idea is that checkpoint frequency increases **during generation**, a better interpretation is that the checkpoint level should be **stage-dependent** or **time-dependent**, for example:

```math
\ell_j(t) \in \{0,1,2\}
```

or, more generally, a monotone policy that moves from

```text
0 → 1 → 2
```

as more output tokens are generated.

This means:

- early stage: low risk / low recovery value → little or no checkpointing;
- later stage: more accumulated KV state and more work to lose → stronger checkpointing.

So conceptually, the optimization is choosing not only *whether* to checkpoint, but also *how aggressively checkpointing should evolve with request progress*.

---

## 5. Affected Job Set

A useful missing notation is the set of requests whose original GPU fails under scenario `ω`:

```math
\tilde{J}(\omega) = \left\{ j \in J : \sum_{r \in \omega} x_{j,r} = 1 \right\}.
```

Because each admitted request is initially assigned to exactly one GPU, this means:

- `j ∈ \tilde{J}(ω)` if and only if its original assigned replica is in the failed set `ω`;
- only these requests need failover recovery;
- requests outside `\tilde{J}(ω)` continue running on their original surviving replica.

This notation is especially helpful for writing the failover-gap constraint more cleanly.

---

## 6. Checkpointing and Recovery Quantities

Checkpoint level affects recovery indirectly through two quantities:

- `S_j^{ckpt}(ℓ_j)` or, more accurately, `S_j^{ckpt}(ℓ_j(t))`:
  recoverable KV-state size stored in host memory;
- `U_j(ℓ_j, ω)` or dynamic version `U_j(ℓ_j(t), ω)`:
  uncovered suffix that must be replayed after failure.

Intuitively:

- stronger checkpointing stores **more recoverable state**,
- stronger checkpointing leaves **less replay work**,
- but stronger checkpointing usually incurs **higher steady-state overhead**.

Therefore, checkpointing affects failover latency through a tradeoff between:

1. **restore cost** from host memory,
2. **replay cost** for lost progress,
3. **ongoing checkpoint overhead** during normal execution.

---

## 7. Optimization Goal

The overall objective is to maximize admitted useful work, measured by output-token goodput:

```math
\max \sum_{j \in J} G_j y_j.
```

This means the system wants to admit as many valuable requests as possible, but only when it can still satisfy routing, capacity, latency, and failover requirements in both normal and failure cases.

---

## 8. Main Constraints

### 8.1 Initial assignment

Each admitted request must be assigned to exactly one initial replica:

```math
\sum_{r \in R} x_{j,r} = y_j, \qquad \forall j.
```

### 8.2 Re-assignment under failures

Under every failure scenario, each admitted request must still be assigned somewhere:

```math
\sum_{r \in R} \tilde{x}_{j,r}(\omega) = y_j, \qquad \forall j,\; \forall \omega \in \Omega_k.
```

A failed replica cannot receive requests:

```math
\tilde{x}_{j,r}(\omega) \le a_r(\omega), \qquad \forall j, r, \omega \in \Omega_k.
```

### 8.3 Checkpoint-domain constraint

A rejected request cannot receive a positive checkpoint level:

```math
\ell_j \le 2 y_j, \qquad \forall j.
```

### 8.4 Capacity constraints

Under each failure scenario, the total assigned prefill load on replica `r` must fit within its prefill capacity:

```math
\sum_j \tilde{x}_{j,r}(\omega) P_j \le C_r^{pre} a_r(\omega) H,
\qquad \forall r, \omega \in \Omega_k.
```

Similarly for decode load:

```math
\sum_j \tilde{x}_{j,r}(\omega) G_j \le C_r^{dec} a_r(\omega) H,
\qquad \forall r, \omega \in \Omega_k.
```

### 8.5 TTFT and TPOT constraints

For TTFT:

```math
\sum_r \tilde{x}_{j,r}(\omega) \frac{P_j}{C_r^{pre}} \le D_j^{ttft},
\qquad \forall j, \omega \in \Omega_k.
```

For TPOT:

```math
\sum_r \tilde{x}_{j,r}(\omega) \frac{G_j}{C_r^{dec}} \le G_j D_j^{tpot},
\qquad \forall j, \omega \in \Omega_k.
```

These ensure that an admitted request still satisfies its latency SLOs even after being re-routed.

### 8.6 Failover-gap constraint

When a request is affected by a failure, its interruption consists of:

- failure detection / failover overhead `T^{det}`,
- checkpoint restore time from host memory,
- replay time for uncovered work,
- time to resume the next decode step.

A cleaner version using the affected-job set `\tilde{J}(ω)` is:

```math
T^{det}
+ \frac{S_j^{ckpt}(\ell_j)}{B^{ld}}
+ \frac{U_j(\ell_j, \omega)}{C^{rep}}
+ \frac{1}{C^{dec}}
\le D_j^{gap},
\qquad \forall j \in \tilde{J}(\omega),\; \forall \omega \in \Omega_k.
```

This says: for every request whose original GPU fails, the total pause before token streaming resumes must stay within its allowed failover gap.

If you want to keep heterogeneity by destination replica, you can also use the replica-specific version:

```math
T^{det}
+ \frac{S_j^{ckpt}(\ell_j)}{B_r^{ld}}
+ \frac{U_j(\ell_j, \omega)}{C_r^{rep}}
+ \frac{1}{C_r^{dec}}
\le D_j^{gap},
\qquad \forall j, r, \omega \text{ with } \tilde{x}_{j,r}(\omega)=1.
```

But if all GPUs are identical, the simplified form is cleaner and better matches your current direction.

---

## 9. Overall Optimization Problem

Putting everything together, the problem can be summarized as follows:

> **Given a set of requests, a set of identical GPU replicas on one machine, shared host memory for KV-cache checkpoints, and an uncertainty set of GPU-failure scenarios, choose admission, initial routing, post-failure routing, and adaptive checkpoint levels so as to maximize output-token goodput, while ensuring capacity limits, TTFT/TPOT SLOs, and bounded failover interruption under every failure scenario.**

A compact mathematical statement is:

```math
\begin{aligned}
\max_{y, x, \tilde{x}, \ell} \quad & \sum_{j \in J} G_j y_j \\
\text{s.t.} \quad
& \sum_r x_{j,r} = y_j, && \forall j \\
& \sum_r \tilde{x}_{j,r}(\omega) = y_j, && \forall j,\; \forall \omega \in \Omega_k \\
& \tilde{x}_{j,r}(\omega) \le a_r(\omega), && \forall j,r,\omega \in \Omega_k \\
& \ell_j \le 2 y_j, && \forall j \\
& \sum_j \tilde{x}_{j,r}(\omega) P_j \le C_r^{pre} a_r(\omega) H, && \forall r,\omega \in \Omega_k \\
& \sum_j \tilde{x}_{j,r}(\omega) G_j \le C_r^{dec} a_r(\omega) H, && \forall r,\omega \in \Omega_k \\
& \sum_r \tilde{x}_{j,r}(\omega) \frac{P_j}{C_r^{pre}} \le D_j^{ttft}, && \forall j,\omega \in \Omega_k \\
& \sum_r \tilde{x}_{j,r}(\omega) \frac{G_j}{C_r^{dec}} \le G_j D_j^{tpot}, && \forall j,\omega \in \Omega_k \\
& T^{det} + \frac{S_j^{ckpt}(\ell_j)}{B^{ld}} + \frac{U_j(\ell_j,\omega)}{C^{rep}} + \frac{1}{C^{dec}} \le D_j^{gap}, && \forall j \in \tilde{J}(\omega),\; \forall \omega \in \Omega_k \\
& y_j, x_{j,r}, \tilde{x}_{j,r}(\omega) \in \{0,1\}, && \forall j,r,\omega \\
& \ell_j \in \{0,1,2\}, && \forall j.
\end{aligned}
```

---

## 10. What Is New in This Formulation

Compared with a standard admission-and-routing formulation, your idea adds two important ingredients:

### Failure-aware robust re-routing

The optimization is not only about normal-case placement. It explicitly requires that admitted requests remain feasible under **any failure scenario with up to `k` failed GPUs**.

### Adaptive checkpointing as a control dimension

Checkpointing is not treated as a fixed implementation detail. Instead, it is elevated into a decision dimension that controls the tradeoff between:

- normal-case overhead,
- recovery-state size,
- replay work,
- failover token gap.

This is the core systems insight of the idea: **checkpointing policy should be co-designed with routing and admission, rather than optimized in isolation**.

---

## 11. Important Modeling Note

Your current simplified optimization writes checkpoint class as a single variable `\ell_j`. That is fine for a first clean formulation.

But your actual idea is stronger:

- checkpoint level is **progress-dependent**,
- it tends to increase as the request generates more tokens,
- so the true policy is closer to a **dynamic checkpoint schedule** than a fixed class.

In the paper, you can present this in two stages:

1. **Base optimization model**: use a single request-level class `ℓ_j ∈ {0,1,2}` for clarity;
2. **System interpretation / extension**: explain that in implementation, the checkpoint level can evolve over time as generation progresses, e.g. from `0` to `1` to `2`.

This keeps the optimization readable while preserving the real intuition of your design.

---

## 12. One-Sentence Summary

This work studies **fault-tolerant single-node multi-GPU LLM serving** where identical GPU replicas periodically checkpoint KV-cache state to shared host memory, and a scheduler jointly optimizes **admission, routing, failover re-routing, and adaptive checkpoint intensity** to maximize goodput while meeting **TTFT, TPOT, and failure-gap SLOs** under bounded GPU failures.
