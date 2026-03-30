# Fault-Tolerant Multi-GPU LLM Serving with Adaptive KV-Cache Checkpointing

## 1. Core Idea

We consider a **single machine with multiple GPU replicas**, where each GPU loads the **same LLM model**. The serving system is **online**: requests arrive continuously, and the scheduler runs at **periodic decision epochs**. At each epoch, it observes the current system snapshot—active requests, newly arrived queued requests, current GPU loads, and checkpoint states—and decides which queued requests to admit and where to dispatch them. During generation, each request periodically checkpoints its KV cache to **shared host memory (RAM)** so that its execution state can survive GPU failures.

When one or more GPUs become unavailable, the scheduler **re-routes affected requests** to surviving GPUs, restores the latest checkpointed KV state from host memory, and resumes generation there.

The key feature is that checkpointing is **adaptive** rather than fixed, but
it is implemented as a **runtime-local online policy** rather than a solver
decision. The current implementation uses the following rule:

- only re-evaluate when a new full KV block becomes stable;
- consider the entire unpublished stable full-block prefix;
- publish that prefix when

```text
Δreplay_saved > Δload + λ * Δcheckpoint_overhead.
```

So the system jointly decides:

1. **which requests to admit**,
2. **where to place them initially**,
3. **how to reassign them after failures**,

while checkpoint publication is handled online by each engine.

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

> **How should the system jointly make routing, admission, and recovery decisions, while using an online adaptive checkpoint policy, so that it remains feasible under failures while still maximizing useful serving throughput?**

---

## 3. Problem Setting

### System architecture

We study a **single-node, multi-GPU replicated serving system**:

- one machine contains a set of GPU replicas `r ∈ R`;
- all replicas load the **same model**;
- host memory is shared and is used to store checkpointed KV-cache state;
- a centralized scheduler dispatches requests and performs failover;
- the system runs online, but the scheduler optimizes at **periodic decision epochs** rather than by solving one global offline problem over the entire lifetime of all requests.

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

We use a short planning horizon `H` for capacity accounting **within each decision epoch**. In other words, `H` is a rolling local horizon used by the controller on the current snapshot, not a once-for-all offline horizon over the full future workload.

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

When we refer to the request set `J` in the optimization below, it should be interpreted as the **current epoch snapshot** `J_t`, not the full future workload. In practice, `J_t` contains (i) requests that are already active in the system and (ii) newly arrived queued requests that are eligible for admission at the current decision epoch.

---

## 4. Decision Variables

### Admission

```math
y_j \in \{0,1\}
```

indicates whether request `j` is admitted **at the current decision epoch**. For newly arrived queued requests, `y_j=1` means the scheduler admits them now. For active requests already running in the system snapshot, admission is already fixed and they remain part of the current state.

### Initial routing / epoch dispatch

```math
x_{j,r} \in \{0,1\}
```

indicates whether request `j` is dispatched to replica `r` **by the current decision epoch**. For newly admitted requests, this is their initial dispatch. For active requests already running, their current placement is part of the snapshot state and can be treated as fixed unless the model explicitly allows migration.

### Re-routing after failures

```math
\tilde{x}_{j,r}(\omega) \in \{0,1\}
```

indicates whether request `j` is assigned to replica `r` under failure scenario `ω` **for the current system snapshot**.

### Runtime online checkpoint policy

Checkpointing is **not** a master-variable in the current implementation.
Instead, each engine applies a local online rule on each active request.

Let:

- `T_j^{stable}(t)` be the number of tokens that already belong to stable
  full KV blocks at time `t`;
- `T_j^{pub}(t)` be the number of tokens already published into the shared
  checkpoint store;
- `ΔT_j(t) = T_j^{stable}(t) - T_j^{pub}(t)` be the unpublished stable suffix.

The runtime re-evaluates only when `T_j^{stable}(t)` increases by one full
block. At that point it estimates:

```math
\Delta \text{replay}_j(t) = \frac{\Delta T_j(t)}{C^{rep}},
```

```math
\Delta \text{load}_j(t) = \frac{\Delta S_j(t)}{B^{ld}},
```

```math
\Delta \text{ckpt}_j(t) = \frac{\Delta S_j(t)}{B^{ckpt}},
```

where `ΔS_j(t)` is the KV size of the unpublished stable suffix.

Then it publishes iff

```math
\Delta \text{replay}_j(t) >
\Delta \text{load}_j(t) + \lambda \, \Delta \text{ckpt}_j(t).
```

Intuitively:

- early in generation, the unpublished stable suffix is small, so replay is cheap;
- later in generation, the unpublished stable suffix grows, so replay becomes expensive;
- publication happens exactly when the saved replay cost is worth the extra
  restore and checkpoint-copy cost.

---

For compactness in the decomposition algorithm, we also use the vector notation

```math
\mathbf y := \{y_j\}_{j\in J},\qquad
\mathbf x := \{x_{j,r}\}_{j\in J,r\in R},
```

and, for each failure scenario,

```math
\tilde{\mathbf x}(\omega) := \{\tilde{x}_{j,r}(\omega)\}_{j\in J,r\in R},\qquad \forall \omega \in \Omega_k.
```


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

The runtime checkpoint policy affects recovery through two quantities:

- `S_j^{pub}(t)`:
  recoverable KV-state size already published to shared host memory;
- `U_j^{tail}(t, \omega)`:
  uncovered suffix that must be replayed after failure.

Intuitively:

- publishing more state stores **more recoverable KV**,
- publishing more state leaves **less replay work**,
- but publishing more state incurs **higher steady-state copy overhead**.

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

### 8.3 Runtime checkpoint trigger

Checkpoint publication is evaluated only for admitted active requests.
For each request `j`, the runtime checks the unpublished stable suffix
`ΔT_j(t)` only when a new full block appears and publishes iff

```math
\frac{\Delta T_j(t)}{C^{rep}}
>
\frac{\Delta S_j(t)}{B^{ld}}
+ \lambda \frac{\Delta S_j(t)}{B^{ckpt}}.
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
+ \frac{S_j^{pub}(t)}{B^{ld}}
+ \frac{U_j^{tail}(t, \omega)}{C^{rep}}
+ \frac{1}{C^{dec}}
\le D_j^{gap},
\qquad \forall j \in \tilde{J}(\omega),\; \forall \omega \in \Omega_k.
```

This says: for every request whose original GPU fails, the total pause before token streaming resumes must stay within its allowed failover gap.

If you want to keep heterogeneity by destination replica, you can also use the replica-specific version:

```math
T^{det}
+ \frac{S_j^{pub}(t)}{B_r^{ld}}
+ \frac{U_j^{tail}(t, \omega)}{C_r^{rep}}
+ \frac{1}{C_r^{dec}}
\le D_j^{gap},
\qquad \forall j, r, \omega \text{ with } \tilde{x}_{j,r}(\omega)=1.
```

But if all GPUs are identical, the simplified form is cleaner and better matches your current direction.

---

## 9. Overall Optimization Problem

Putting everything together, the problem can be summarized as follows:

> **At each decision epoch, given the current scheduling snapshot consisting of active requests, newly arrived queued requests, current replica loads, shared host-memory checkpoint state, and an uncertainty set of GPU-failure scenarios, choose admission, epoch dispatch, and post-failure routing so as to maximize output-token goodput, while a runtime online checkpoint policy maintains recoverable state and all capacity, TTFT/TPOT, and failover-gap constraints remain satisfied under every failure scenario.**

A compact mathematical statement is:

```math
\begin{aligned}
\max_{y, x, \tilde{x}} \quad & \sum_{j \in J} G_j y_j \\
\text{s.t.} \quad
& \sum_r x_{j,r} = y_j, && \forall j \\
& \sum_r \tilde{x}_{j,r}(\omega) = y_j, && \forall j,\; \forall \omega \in \Omega_k \\
& \tilde{x}_{j,r}(\omega) \le a_r(\omega), && \forall j,r,\omega \in \Omega_k \\
& \sum_j \tilde{x}_{j,r}(\omega) P_j \le C_r^{pre} a_r(\omega) H, && \forall r,\omega \in \Omega_k \\
& \sum_j \tilde{x}_{j,r}(\omega) G_j \le C_r^{dec} a_r(\omega) H, && \forall r,\omega \in \Omega_k \\
& \sum_r \tilde{x}_{j,r}(\omega) \frac{P_j}{C_r^{pre}} \le D_j^{ttft}, && \forall j,\omega \in \Omega_k \\
& \sum_r \tilde{x}_{j,r}(\omega) \frac{G_j}{C_r^{dec}} \le G_j D_j^{tpot}, && \forall j,\omega \in \Omega_k \\
& T^{det} + \frac{S_j^{pub}(t)}{B^{ld}} + \frac{U_j^{tail}(t,\omega)}{C^{rep}} + \frac{1}{C^{dec}} \le D_j^{gap}, && \forall j \in \tilde{J}(\omega),\; \forall \omega \in \Omega_k \\
& y_j, x_{j,r}, \tilde{x}_{j,r}(\omega) \in \{0,1\}, && \forall j,r,\omega \\
\end{aligned}
```

---

## 10. Benders-Style Decomposition with CRP-Inspired Pooled Recovery Screening

The robust formulation above is useful only if we also provide a structured way to solve it. A natural interpretation of the problem is a **master-plus-recourse** decomposition:

- the **master problem** decides the current-epoch controls, namely admission `y` and epoch dispatch `x` on the current system snapshot;
- for each failure scenario `\omega \in \Omega_k`, a **recovery subproblem** checks whether the affected requests can be re-routed and resumed on surviving replicas while still respecting memory, compute, and latency constraints.

Checkpointing is not optimized by the master. Instead, the solver treats the
current runtime-published checkpoint state as a fixed input when estimating
recovery cost.

This is closest to a **Benders-style scenario-guided decomposition**: the master proposes a candidate solution, each scenario subproblem either produces a feasible recovery plan or returns an infeasibility certificate, and that certificate is turned into a new cut for the next master iteration. If later we replace the full scenario sweep by a worst-case scenario oracle, the same structure can be tightened toward a more CCG-like variant.

### 10.1 Recovery-side pooled screen

Under a fixed failure scenario `\omega`, let

```math
R_\omega = R \setminus \omega
```

be the surviving replicas, and let

```math
u_r(\omega)
```

denote the **residual recovery budget** of surviving replica `r`, i.e. the amount of additional recovery work that `r` can still absorb after accounting for the work that remains on it.

We then define the total recovery work induced by affected requests as

```math
W(\omega) = \sum_{j \in \tilde{J}(\omega)}
\left(
\frac{S_j^{pub}(t)}{B^{ld}} +
\frac{U_j^{tail}(t,\omega)}{C^{rep}}
\right),
```

and the aggregate pooled recovery headroom as

```math
H_{\text{pool}}(\omega) = \sum_{r \in R_\omega} u_r(\omega).
```

This gives a **CRP-inspired pooled screening rule**:

- if `W(\omega) > H_{\text{pool}}(\omega)`, then the scenario is immediately infeasible;
- otherwise, we continue with an exact assignment-level recovery check.

The point of this screen is not to claim a full CRP theorem for our system, but to exploit the same intuition: with identical GPUs and shared host memory, the surviving replicas can often be treated as one pooled recovery resource for a first-pass aggregate feasibility test.

### 10.2 Exact recovery by flow

When the pooled screen passes, we still need an exact assignment because aggregate capacity alone does not guarantee that every affected request can be placed on some surviving GPU. We therefore build a bipartite graph between affected requests `\tilde{J}(\omega)` and surviving replicas `R_\omega`:

- a request node `j` is connected to replica `r` only if assigning `j` to `r` is individually feasible under memory, compute, TTFT/TPOT, and failover-gap constraints;
- replica-side capacities are derived from residual budgets `u_r(\omega)`;
- a max-flow or min-cost max-flow then checks whether all affected requests can be covered.

Using **min-cost max-flow** is helpful when we want the recovery plan to prefer lower replay overhead, lower failover delay, or better load balance. If we only care about feasibility, the same step can be viewed as a plain max-flow.

### 10.3 Algorithms

The following algorithms instantiate this decomposition. In the current
implementation, checkpoint publication is handled by the runtime online rule,
so the master optimizes only admission/placement and the recovery subproblem
consumes the resulting published checkpoint state as a fixed input.

```latex
\begin{algorithm}[t]
\caption{Benders-Style Robust Serving with Pooled Recovery Screening}
\begin{algorithmic}[1]
\Require request set $J$, replica set $R$, failure scenario set $\Omega_k$
\Ensure admission decisions $y$, initial placement $x$, recovery plans $\tilde{x}(\omega)$ for all $\omega \in \Omega_k$
\State initialize cut set $\mathcal{C} \gets \emptyset$
\While{true}
    \State solve the master problem with cuts $\mathcal{C}$
    \State obtain candidate solution $(y,x)$
    \State feasible $\gets$ True
    \ForAll{$\omega \in \Omega_k$}
        \State $(\tilde{x}(\omega), \textsc{status}, \textsc{cert}) \gets \textsc{Pooled-Flow-Recovery}(y,x,\omega)$
        \If{$\textsc{status} = \textsc{Infeasible}$}
            \State generate a logic-based scenario cut from $(y,x,\omega,\textsc{cert})$
            \State $\mathcal{C} \gets \mathcal{C} \cup \{\textsc{cut}(y,x,\omega,\textsc{cert})\}$
            \State feasible $\gets$ False
            \State break
        \EndIf
    \EndFor
    \If{feasible}
        \State \Return $(y,x,\{\tilde{x}(\omega)\}_{\omega \in \Omega_k})$
    \EndIf
\EndWhile
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[t]
\caption{\textsc{Pooled-Flow-Recovery} under scenario $\omega$}
\begin{algorithmic}[1]
\Require candidate solution $(y,x)$, failure scenario $\omega$
\Ensure rerouting plan $\tilde{x}(\omega)$ or \textsc{Infeasible} with certificate $\textsc{cert}$
\State $R_\omega \gets R \setminus \omega$
\State $\tilde{J}(\omega) \gets \{\,j\in J : \sum_{r\in\omega} x_{j,r}=1\,\}$
\State compute residual recovery budgets $u_r(\omega)$ for all $r \in R_\omega$
\State compute per-request recovery work $w_j(\omega)$ for all $j \in \tilde{J}(\omega)$
\State compute $W(\omega)$ and $H_{\mathrm{pool}}(\omega)$
\If{$W(\omega) > H_{\mathrm{pool}}(\omega)$}
    \State extract a minimal overloaded subset $Q \subseteq \tilde{J}(\omega)$
    \State $\textsc{cert} \gets (\textsc{PoolOverload}, Q, W(\omega)-H_{\mathrm{pool}}(\omega))$
    \State \Return \textsc{Infeasible}, $\textsc{cert}$
\EndIf
\State build a bipartite graph on $\tilde{J}(\omega)$ and $R_\omega$
\State add edge $(j,r)$ only if assigning $j$ to $r$ is feasible under memory, compute, TTFT/TPOT, and failover-gap constraints
\State set replica capacities from residual budgets $u_r(\omega)$
\State solve a min-cost max-flow
\If{not all requests in $\tilde{J}(\omega)$ are covered}
    \State extract a deficient request subset $S$ from the min-cut / max-flow residual graph
    \State $\textsc{cert} \gets (\textsc{HallDeficit}, S)$
    \State \Return \textsc{Infeasible}, $\textsc{cert}$
\EndIf
\State recover $\tilde{x}(\omega)$ from the flow solution
\State \Return $\tilde{x}(\omega), \textsc{Feasible}, \emptyset$
\end{algorithmic}
\end{algorithm}
```

These algorithms do two useful things for the overall paper story:

1. they turn the formulation into a **solver-backed optimization problem** rather than a bare model;  
2. they make the recovery side more interpretable by separating an **aggregate pooled screen** from an **exact combinatorial recovery assignment**.

---


### 10.4 What the certificate and cut can be

The recovery subproblem is combinatorial, so the cleanest interpretation is **logic-based Benders** rather than classical dual Benders. In other words, the subproblem does not need to return a linear-program dual ray; it only needs to return a certificate that explains *why* the current master solution fails under scenario `\omega`, and the master then adds a cut that prevents the same infeasible pattern from reappearing.

A practical way to organize this is to use two certificate types.

#### (a) Pooled-overload certificate

Define the per-request recovery work

```math
w_j(\omega,t) := \frac{S_j^{pub}(t)}{B^{ld}} + \frac{U_j^{tail}(t,\omega)}{C^{rep}}.
```

If the pooled screen fails, i.e.

```math
\sum_{j\in \tilde J(\omega)} w_j(\omega,t) > H_{\text{pool}}(\omega),
```

then the subproblem can return a certificate of the form

```text
cert = (PoolOverload, ω, Q, Δ)
```

where:

- `Q ⊆ \tilde J(\omega)` is a minimal overloaded subset whose total recovery work already exceeds the pooled headroom;
- `Δ > 0` is the overload margin.

A simple way to obtain `Q` is to sort affected requests by `w_j(\omega,t)` and keep adding them until the partial sum first exceeds `H_{\text{pool}}(\omega)`.

#### (b) Flow-deficit / Hall-deficit certificate

If the pooled screen passes but the flow still fails to cover all affected requests, then the issue is no longer aggregate work but **assignment structure**: some subset of requests can only go to a too-small set of surviving replicas.

The subproblem can then return a certificate of the form

```text
cert = (HallDeficit, ω, S)
```

where `S ⊆ \tilde J(\omega)` is a deficient request subset extracted from the residual graph or min-cut. Intuitively, `S` is a set of requests whose feasible neighboring replicas do not have enough residual capacity to absorb them all.

---

### 10.5 Baseline logic-based cut

For a first implementation, the safest baseline is an **incumbent no-good cut**. This does not try to derive the strongest possible cut; it only guarantees that the next master iteration must change at least one decision involved in the current infeasible pattern.

Using the combined routing variables, a generic no-good cut for an infeasible incumbent `(y^\star, x^\star)` is:

```math
\sum_{j: y_j^\star = 1} (1-y_j)
+ \sum_{j: y_j^\star = 0} y_j
+ \sum_{(j,r): x_{j,r}^\star = 1} (1-x_{j,r})
\ge 1.
```

This cut simply says: the master cannot repeat the exact same admission and placement pattern that was just proven infeasible.

This is **valid but weak**. It is enough to make the algorithm well-defined in the idea stage, and later you can replace it with stronger certificate-aware cuts.

---

### 10.6 Stronger cut directions (later refinement)

Once the basic logic-based version is in place, the certificate can be used to derive stronger families of cuts.

- From a **pooled-overload certificate** `Q`, we want a cut that says the master cannot again place all jobs in `Q` into a configuration that makes them simultaneously vulnerable to `\omega` *unless* it changes admission or checkpoint intensity enough to reduce the recovery work.
- From a **Hall-deficit certificate** `S`, we want a cut that says the master cannot again create the same structurally unrecoverable placement pattern for the requests in `S`.

In the paper draft, I would present these stronger cuts as an extension direction rather than pretending they are already fully derived. That keeps the story honest:

- **current algorithm:** logic-based Benders with explicit certificates;
- **next refinement:** replace no-good cuts by conflict-set cuts that exploit `Q` or `S`.

This is usually a much safer presentation than writing an aggressive closed-form Benders cut too early and then having to defend its validity.




## 10.7 Solver Instantiation for a First Prototype

The sections above are enough for a paper-level method description, but a first implementation needs a **single concrete instantiation** of every quantity. The following choices are deliberately conservative: they give a version that is straightforward to code first, and can later be strengthened.

### 10.7.1 Prototype assumptions

For the first prototype, fix the following modeling choices.

1. **Identical GPUs.** All replicas have the same profiled capacities, so we write `C^{pre}`, `C^{dec}`, `B^{ld}`, and `M^{cap}` without replica index unless heterogeneity is explicitly needed later.
2. **Periodic snapshot controller.** The serving system is online, but at each decision epoch the controller optimizes over a finite snapshot `J_t`, consisting of active requests plus newly arrived queued requests. This is a rolling snapshot-based online controller, not a once-for-all offline planner.
3. **Single-failure baseline.** Start with `k = 1`, so `\Omega_1 = \\{\\{r\\}: r \\in R\\}`. Multi-failure support can be added after the single-failure version works.
4. **Runtime-side checkpointing.** The solver does not optimize checkpoint
   class. Each request carries a currently published checkpoint state from
   the online runtime policy, and the solver consumes that state as a fixed
   input to recovery-cost estimation.
5. **Lookup-table cost model.** All per-request quantities used by the solver are precomputed from profiling tables and then treated as constants inside the optimization.
6. **Decode-side recovery budget.** In the first solver, the pooled headroom `u_r(\omega)` is measured in **time budget on the decode/recovery path**. Memory is still checked separately as an edge-feasibility constraint, but it is not folded into `u_r(\omega)`.
7. **Feasibility first, optimization second.** In phase 1, the exact recovery checker is implemented as a small 0-1 feasibility ILP. The earlier flow view remains the conceptual explanation; if later we discretize capacities, the same checker can be reimplemented as min-cost flow.

These assumptions are not the only possible choices; they are just the cleanest way to make the method immediately implementable.

### 10.7.2 Profiled tables and units

To make the model executable, every symbolic quantity should be backed by a
profiled constant. In the current implementation, the important profiled
constants are:

- `p_j`: normal-case prefill time demand within the planning horizon;
- `d_j`: normal-case decode time demand within the planning horizon;
- `B^{ckpt}`: GPU→Host checkpoint bandwidth;
- `B^{ld}`: Host→GPU restore bandwidth;
- `C^{rep}`: replay throughput;
- `o_{j,m}^{ckpt}`, `s_{j,m}^{ld}`, `u_{j,m}^{rep}`, `m_{j,m}^{rec}`:
  fixed solver-side cost lookups under checkpoint stage `m`.

In code, these values should come from lookup tables indexed by a request bucket, for example:

```text
bucket(j) = (prompt_length_bucket, output_length_bucket, progress_bucket)
```

Then `o_{j,m}^{ckpt}`, `s_{j,m}^{ld}`, `u_{j,m}^{rep}`, and `m_{j,m}^{rec}` are table lookups. In the first prototype, it is completely fine to use a single fixed progress bucket per request snapshot.

The important implementation rule is:

> **Every quantity appearing in the master or recovery checker must be a number before the solver starts.**

That means the first prototype should not try to optimize over an implicit
checkpoint policy inside the solver. Instead, profile or estimate the needed
bandwidth / throughput constants, let the runtime produce checkpoint state
online, and let the solver consume that state through fixed cost lookups.

### 10.7.3 Linearized master variables

To write the master problem directly in code, it is cleaner to use a single
binary variable that combines placement with the fixed checkpoint stage used
for cost lookup:

```math
z_{j,r,m} \in \{0,1\}, \qquad j \in J,\ r \in R,\ m \in \{0,1,2\}.
```

Interpretation:

- `z_{j,r,m} = 1` means request `j` is admitted, initially placed on replica
  `r`, and the solver evaluates it under fixed checkpoint stage `m`.

Then the earlier variables can be recovered by:

```math
y_j = \sum_{r \in R} \sum_{m=0}^2 z_{j,r,m},
```

```math
x_{j,r} = \sum_{m=0}^2 z_{j,r,m},
```

This avoids bilinear products in the master even though checkpointing itself
is no longer a solver decision.

### 10.7.4 Concrete master problem for code

A first implementation can use the following master MIP **at each decision epoch on the current snapshot `J_t`**:

```math
\max \sum_{j \in J} G_j y_j
```

subject to

```math
\sum_{r \in R} \sum_{m=0}^2 z_{j,r,m} \le 1, \qquad \forall j,
```

```math
y_j = \sum_{r \in R} \sum_{m=0}^2 z_{j,r,m}, \qquad \forall j,
```

normal-case prefill capacity:

```math
\sum_{j \in J} \sum_{m=0}^2 p_j z_{j,r,m} \le H^{pre}, \qquad \forall r,
```

normal-case decode plus checkpoint-overhead capacity:

```math
\sum_{j \in J} \sum_{m=0}^2 \bigl(d_j + o_{j,m}^{ckpt}\bigr) z_{j,r,m} \le H^{dec}, \qquad \forall r,
```

optional normal-case memory capacity:

```math
\sum_{j \in J} \sum_{m=0}^2 m_{j,m}^{run} z_{j,r,m} \le M^{cap}, \qquad \forall r,
```

and optional per-request normal-case SLO filters:

```math
z_{j,r,m} = 0 \quad \text{if request } j \text{ cannot meet normal-case TTFT/TPOT on replica } r.
```

Here:

- `H^{pre}` is the available prefill-time budget on one replica over the planning horizon;
- `H^{dec}` is the available decode-time budget on one replica over the planning horizon;
- `m_{j,m}^{run}` is the normal-case running memory footprint of request `j`
  under checkpoint stage `m`.

This master is intentionally simple. It does **not** yet encode all failure scenarios explicitly; failure robustness is enforced by the Benders-style cuts returned by the recovery checker.

### 10.7.5 Exact definition of affected requests and per-request recovery work

Given a candidate master solution `z`, define the current assigned replica and
fixed checkpoint-stage tag of each admitted or active request represented in
the snapshot as the unique pair `(r_j^0, m_j^0)` with `z_{j,r_j^0,m_j^0} = 1`.

Under scenario `\omega`, the affected set is:

```math
\tilde J(\omega) = \{ j \in J : r_j^0 \in \omega \}.
```

For each affected request, define its recovery work in **seconds of recovery-path time** as:

```math
w_j(\omega) = s_{j,m_j^0}^{ld} + u_{j,m_j^0}^{rep}.
```

In the first prototype, `w_j(\omega)` is independent of the destination GPU because GPUs are identical. If later the destination matters, replace it by `w_{j,r}(\omega)`.

### 10.7.6 Exact definition of residual recovery budget `u_r(\omega)`

For a surviving replica `r \in R_\omega`, let `J_r^{surv}(\omega)` be the set of unaffected requests that were already placed on `r` in the master solution:

```math
J_r^{surv}(\omega) = \{ j \in J \setminus \tilde J(\omega) : r_j^0 = r \}.
```

The first prototype measures headroom only on the decode/recovery path. Define

```math
u_r(\omega) = \max\Bigl\{0,\ H^{dec} - \sum_{j \in J_r^{surv}(\omega)} \bigl(d_j + o_{j,m_j^0}^{ckpt}\bigr)\Bigr\}.
```

Interpretation:

- `H^{dec}` is the total decode-side time budget of replica `r` over the planning horizon;
- unaffected jobs already occupying `r` consume part of that budget;
- the leftover budget `u_r(\omega)` is what remains available for replay-and-resume work of failover jobs.

Then the pooled headroom is exactly

```math
H_{\text{pool}}(\omega) = \sum_{r \in R_\omega} u_r(\omega).
```

This is the quantity used in the pooled screen.

### 10.7.7 Destination-memory availability

Because `u_r(\omega)` only captures decode/recovery time headroom, the recovery checker should also compute destination free memory:

```math
M_r^{free}(\omega) = M^{cap} - \sum_{j \in J_r^{surv}(\omega)} m_{j,m_j^0}^{run}.
```

If memory availability is the dominant bottleneck in your real system, you can later strengthen the pooled screen by replacing time-only headroom with

```math
u_r^{time}(\omega),\qquad M_r^{free}(\omega),
```

and screening with both aggregate time and aggregate memory. But for the first implementation, it is simpler to keep memory in the exact checker only.

### 10.7.8 Explicit edge-feasibility predicate

The recovery checker should create edge `(j,r)` only if the predicate `F_{j,r}(\omega) = 1` holds. In the first prototype:

```math
F_{j,r}(\omega) = 1
```

if and only if all four conditions below are satisfied.

1. **Replica survives:** `r \notin \omega`.
2. **Memory fits:**

```math
m_{j,m_j^0}^{rec} \le M_r^{free}(\omega).
```

3. **Per-request failover-gap bound holds:**

```math
T^{det} + w_j(\omega) + \frac{1}{C^{dec}} \le D_j^{gap}.
```

4. **Destination-specific SLO filter holds:** the profiled deterministic TTFT/TPOT bound of resuming `j` on `r` is acceptable.

In the first prototype, condition 4 can be implemented by a boolean lookup `slo_ok[j,r]`. If GPUs are identical, then `slo_ok[j,r]` is the same for every surviving `r`, and you can collapse it to `slo_ok[j]`.

### 10.7.9 Exact recovery checker for code

For code, the cleanest exact checker is a **small assignment ILP** over affected requests and surviving replicas. Introduce binary variables

```math
a_{j,r}(\omega) \in \{0,1\}, \qquad j \in \tilde J(\omega),\ r \in R_\omega.
```

Interpretation:

- `a_{j,r}(\omega) = 1` means affected request `j` is recovered onto surviving replica `r`.

Then solve the feasibility ILP:

```math
\sum_{r \in R_\omega} a_{j,r}(\omega) = 1, \qquad \forall j \in \tilde J(\omega),
```

```math
a_{j,r}(\omega) \le F_{j,r}(\omega), \qquad \forall j \in \tilde J(\omega),\ r \in R_\omega,
```

```math
\sum_{j \in \tilde J(\omega)} w_j(\omega) a_{j,r}(\omega) \le u_r(\omega), \qquad \forall r \in R_\omega,
```

```math
\sum_{j \in \tilde J(\omega)} m_{j,m_j^0}^{rec} a_{j,r}(\omega) \le M_r^{free}(\omega), \qquad \forall r \in R_\omega.
```

If this ILP is feasible, recover the rerouting plan by setting

```math
\tilde x_{j,r}(\omega) = a_{j,r}(\omega) \quad \text{for } j \in \tilde J(\omega),
```

and keep unaffected requests on their original surviving replica.

If you want an objective, the first safe choice is simply:

```math
\min 0.
```

That is, implement it as pure feasibility first. After that works, you can add a cost such as

```math
c_{j,r}(\omega) = \lambda_r(\omega),
```

where `\lambda_r(\omega)` is the current normalized load of destination replica `r`, so the checker prefers less-loaded recovery targets.

### 10.7.10 Relationship to the earlier flow description

The earlier text described the exact checker as bipartite min-cost max-flow. That view is still correct at the level of intuition, but **for code** the assignment ILP above is less ambiguous because requests have non-unit recovery work `w_j(\omega)` and non-unit memory demand `m_{j,m_j^0}^{rec}`.

So the implementation recommendation is:

- **paper story:** pooled screen + flow-style exact recovery;
- **first code version:** pooled screen + weighted assignment ILP;
- **later optimization:** replace the ILP by a true min-cost flow only after discretizing replica capacities or simplifying recovery work to unit demand.

### 10.7.11 Certificates and cuts in the first code version

For code, the simplest robust loop is:

- use the pooled-overload certificate when `W(\omega) > H_{\text{pool}}(\omega)`;
- otherwise solve the exact recovery ILP;
- if the ILP is infeasible, fall back to the incumbent no-good cut.

This means the first code version only needs two concrete certificate extractors.

#### Pooled-overload certificate extraction

Given affected requests and recovery works `w_j(\omega)`, sort requests in descending order of `w_j(\omega)` and greedily accumulate until the partial sum first exceeds `H_{\text{pool}}(\omega)`. Return that subset as `Q`.

Pseudo-rule:

```text
sort affected jobs by w_j descending
sum = 0, Q = []
for j in sorted jobs:
    Q.append(j)
    sum += w_j
    if sum > H_pool(ω):
        break
return (PoolOverload, ω, Q, sum - H_pool(ω))
```

#### Assignment-failure certificate extraction

If the exact recovery ILP is infeasible, the first implementation does **not** need to extract a Hall-deficit set immediately. It can simply return

```text
(NoAssignment, ω)
```

and add the incumbent no-good cut. This is weaker than a structural Hall-deficit cut, but it is much easier to implement correctly.

So, for code, the recommendation is:

- keep `PoolOverload` as the only explicit structured certificate in phase 1;
- treat `HallDeficit` as an optional strengthening once the basic solver is already running.

### 10.7.12 Scenario generation and solve loop

For a first end-to-end implementation, use the following order **at each decision epoch**.

1. Build the current snapshot `J_t` from active requests and newly arrived queued requests.
2. Enumerate all single-GPU failures:

```math
\Omega_1 = \{\{r\}: r \in R\}.
```

3. Start the epoch master with no cuts.
4. Solve the master on the current snapshot.
5. For each singleton failure scenario, run the pooled screen and then the exact recovery ILP if needed.
6. If all scenarios are feasible, commit the epoch decisions.
7. Otherwise add one cut and repeat on the same snapshot.

This is intentionally a **full scenario sweep** for `k = 1`. It is simple to code and easy to debug. Only after that should you move to larger `k`, sampled scenarios, or adversarial scenario generation.

### 10.7.13 Minimal implementation roadmap

A reader should be able to code the first solver by following this order:

1. profile request buckets and produce lookup tables for `p_j`, `d_j`, `o_{j,m}^{ckpt}`, `s_{j,m}^{ld}`, `u_{j,m}^{rep}`, `m_{j,m}^{run}`, and `m_{j,m}^{rec}`;
2. at each decision epoch, build the current snapshot `J_t` from active requests and newly arrived queued requests;
3. implement the epoch master MIP using `z_{j,r,m}`;
4. given a master solution, reconstruct `(r_j^0, m_j^0)` for each admitted or active request represented in the snapshot;
5. enumerate singleton failures and compute `\tilde J(\omega)`, `w_j(\omega)`, `u_r(\omega)`, and `M_r^{free}(\omega)`;
6. run the pooled screen;
7. if needed, solve the recovery ILP for that scenario;
8. if any scenario fails, add the no-good cut and repeat on the same snapshot.

If the goal is just to make the method runnable, this is enough. All stronger cut families, dynamic checkpoint schedules, and multi-failure scenario generation can be layered on later.


## 11. What Is New in This Formulation

Compared with a standard admission-and-routing formulation, your idea adds two important ingredients:

### Failure-aware robust re-routing

The optimization is not only about normal-case placement. It explicitly requires that admitted requests remain feasible under **any failure scenario with up to `k` failed GPUs**.

### Online checkpointing as a coordinated runtime policy

Checkpointing is not treated as a fixed implementation detail. Instead, it is
handled by a runtime online policy that is coordinated with the solver-side
admission / routing decisions and controls the tradeoff between:

- normal-case overhead,
- recovery-state size,
- replay work,
- failover token gap.

This is the core systems insight of the implemented design:
**checkpointing policy should be co-designed with routing and admission, but
executed online in the runtime rather than optimized as a separate master
variable**.

---

## 12. Important Modeling Note

The current implementation does **not** optimize checkpoint class as a master
variable.

Instead:

1. the solver optimizes admission, placement, and recovery feasibility on the
   current snapshot;
2. the runtime uses a block-granular online checkpoint rule and publishes the
   current unpublished stable prefix when
   `Δreplay_saved > Δload + λ * Δcheckpoint_overhead`.

This keeps the optimization readable while matching the actual system
behavior more faithfully.

---

## 13. One-Sentence Summary

This work studies **fault-tolerant single-node multi-GPU LLM serving** where
identical GPU replicas periodically checkpoint KV-cache state to shared host
memory, and an **online scheduler with periodic decision epochs** optimizes
**admission, epoch dispatch, and failover re-routing** on each current system
snapshot to maximize goodput while a **runtime online checkpoint heuristic**
decides when to publish additional KV state, all while meeting **TTFT, TPOT,
and failure-gap SLOs** under bounded GPU failures using a
**Benders-style decomposition** with **CRP-inspired pooled recovery
screening** and **flow-based recovery assignment**.
