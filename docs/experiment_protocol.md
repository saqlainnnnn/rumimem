# Experiment protocol (initial draft)

## Primary metric

Delayed recall accuracy after a specified number of intervening distractor sessions.

## Secondary metrics

- Recall accuracy by distractor count
- Correction accuracy
- Stale-memory error rate
- Interference on previously stored facts
- Number of successful associations versus memory size
- Update and read latency
- Checkpoint size and restore correctness
- Cross-user leakage rate (must be zero in controlled tests)

## Required baselines

- No persistent memory
- Neural fast-weight memory
- Memory with momentum disabled
- Memory with forgetting disabled
- Shuffled/random write order
- Text-retrieval baseline (comparison only, not the primary contribution)

## Reproducibility checklist

Record random seed, commit hash, config, dependency versions, device, wall-clock time, training steps, checkpoint path, and all metrics. Never select the best checkpoint using the final test set.
