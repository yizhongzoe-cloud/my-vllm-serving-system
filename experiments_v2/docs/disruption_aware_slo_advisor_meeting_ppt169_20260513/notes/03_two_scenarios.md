[Transition] Two scenarios — engine failure and priority preemption — share one root cause: there's no cheap way to resume a partial decode.
Engine failure: a long request lives on one GPU, that GPU disappears (hardware fault, OOM, drain, kill, multi-tenant preempt). Long requests are disproportionately exposed because their lifetime scales with disruption probability. Today's recovery is full reprefill on a surviving engine.
Priority preempt: a tight-SLO request arrives, deadline tighter than the running request's. Schedulers today refuse to preempt long-context decode because reprefill cost dominates priority benefit, so the tight request waits or misses. Same root cause: no cheap resume.

Key points: ① two scenarios, one root cause ② long requests overexposed in both ③ "no cheap resume" is the unifying mechanism gap
Duration: 2.5 minutes