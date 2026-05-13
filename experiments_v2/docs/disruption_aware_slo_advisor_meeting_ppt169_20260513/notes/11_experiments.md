[Transition] Three experiments, one per contribution. Plots are placeholders for now.
E_M1 — main figure: SLO attainment versus QPS, three systems. Claim: ours holds attainment past the saturation knee where baselines drop below 90%. Long-context gap is widest because each preempt saves seconds of prefill the baselines redo. We have 1.0–8.0 QPS, 3 seeds on A6000; L40S replicate in progress.
E_M4 — picker ablation: same axes plus ours_no_picker. Claim: ours_no_picker lies between ours and reroute_no_ckpt — mechanism alone gives passive benefit, picker actively realizes the SLO advantage. Two contributions independent and additive. ours_no_picker runs scheduled this week.
E_D1 — failover gap: SIGKILL to next-token latency versus context length. Claim: ours about 1.5 s on RULER 16K, no_ckpt about 10–20 s at 16K, gap grows with context length, tens of seconds extrapolated at 64K. Demo works on RULER 16K; scaling to longer context next.

Key points: ① three experiments, one per contribution ② E_M1 mostly in, E_M4 + E_D1 in progress ③ failover gap grows with context length — wins exactly where we target
Duration: 3.5 minutes