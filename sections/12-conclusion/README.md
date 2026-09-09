<!--
SECTION-CONTRACT
id: 12-conclusion
role: Close the article with durable engineering takeaways.
sources: All preceding sections.
edit_scope: No new measurements or uncited superlatives.
-->

# 12. Conclusion

The path from PR #16 to PR #44 changes the unit of optimization. It begins with
individual Linear layers and ends with the full lifecycle of an adapted,
distributed, quantized video-and-audio generator.

Three conclusions survive across every case:

1. **Optimize real boundaries.** Linear GEMM, attention, communication, and
   decoding have different data contracts. Keeping them explicit enables
   fusion without hiding precision changes.
2. **Treat quality as a measured systems constraint.** Dense islands,
   attention-equivalent smoothing, tensor error, and generated media determine
   which fast profile is usable.
3. **Measure the lifecycle, not only the kernel.** Online conversion, spawn
   semantics, cache ownership, adapter merge order, phase overlap, and offload
   policy can dominate the deployment result.

The final TeleFuser path combines online FP8 Linear, FP8-native Sol attention,
Ulysses sequence parallelism, fused K/V smoothing, and adapter-aware weight
materialization. Its value is not a single universal speedup. It is a
reproducible method for moving a large generative model toward a better
quality-throughput-memory frontier while keeping every trade-off visible.
