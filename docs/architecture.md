# Architecture notes

## Scope

RumiMem investigates fast-weight neural memory for text LLMs. The first milestone is a standalone memory network; LLM integration is intentionally deferred.

## Titans reference update

For key `k_t`, target `v_t`, and memory parameters `Theta_{t-1}`:

```text
loss_t = ||M_{Theta_{t-1}}(k_t) - v_t||^2
g_t    = grad_Theta loss_t
S_t    = eta_t * S_{t-1} - theta_t * g_t
Theta_t = (1 - alpha_t) * Theta_{t-1} + S_t
```

The paper's gates are data-dependent in its main formulation. Fixed gates, simplified optimizers, or different write policies must be labeled as ablations rather than described as an exact reproduction.

## Implementation order

1. Implement the sequential update as the correctness reference.
2. Test the update equations against hand-computable toy cases.
3. Test read-without-write semantics.
4. Test state serialization and exact restore.
5. Add synthetic episodes with delayed recall and distractors.
6. Add ablations and baselines before integrating an LLM.

## Open design decisions

- How are the gates parameterized and trained in the first experiment?
- Which hidden state becomes the value target?
- How do delayed recall gradients influence the writer when updates are unrolled?
- What is the correction protocol, and how is stale-memory error measured?
