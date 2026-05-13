[Transition] Let me start with why we keep coming back to the prefill-throwaway problem.
On our setup, RULER 16K prefill takes about 2.7 seconds, while ShareGPT chat TTFT P95 is 0.45 seconds — roughly 6× gap. So in the long-context regime, every time we discard a prefill and recompute, we burn seconds of GPU time per request — time that could have served someone else. Two distinct events keep forcing us to discard prefill, and they share the same root cause; that's the next slide.

Key points: ① concrete cost (2.7 s vs 0.45 s) ② ~6× gap drives all downstream design ③ throwaway is the wedge, not just "prefill is slow"
Duration: 2 minutes