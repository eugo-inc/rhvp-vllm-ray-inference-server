# SPDX-License-Identifier: Apache-2.0
"""Compare two inference-server images on the same model, the same GPU and the same requests.

A comparison is a list of rounds. Each round brings one side up, times its start, loads it with GuideLLM at a few
concurrency levels, reads its GPU power, and takes it down. The rounds run one at a time, in ABBA order, and every
metric gets a paired verdict (FASTER, TIE, SLOWER or INCONCLUSIVE) from per-pair ratios, an exact test and a
correction for the number of metrics, against a tie band declared before the first round.

Modules:

- plan: the config read and checked, the round order, the metric family; written into the results file first
- runner: the round loop, independent of the platform; writes results.json and results.md after every round
- kube, dockerapi: the two platforms (OpenShift through the Kubernetes API; one machine through the Docker API)
- probe: time to ready and time to first answer
- guidellm: the GuideLLM command line and its report
- logs: the start-up breakdown and the version from the server's own log
- energy: GPU power (DCGM through the cluster's monitoring, or nvidia-smi) and tokens per joule
- analyze: the verdicts, from the rounds in a results file (the adapter boundary: anything that reads a results file
  and writes `analysis` the same way can take its place)
- stats: the exact tests, the corrections and the verdict rule; standard library only
- report: the short markdown table
- show, words: the comparison notebook's display and its fixed texts

Run it with `python -m rhvp_compare --help`.
"""
