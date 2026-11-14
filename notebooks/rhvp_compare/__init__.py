# SPDX-License-Identifier: Apache-2.0
"""Compare two inference-server images on the same model, the same GPU and the same requests.

A comparison is a list of launches. Each launch brings one side up, times its start, loads it with GuideLLM at a
few concurrency levels, reads its GPU power, and takes it down. A round launches every side and its twin (an
identical copy of the side) once, A, A~same, B, B~same, in an order that rotates from round to round. Every case
gets a paired verdict from the per-round ratios B/A and an exact sign test (corrected with Holm's method over the
primary cases that have a p-value), against a bar the twins measured: FASTER, SLOWER, BETTER, WORSE, TIE, UNRESOLVED or
NO VERDICT.

Modules:

- plan: the config read and checked, the lineup and run order, the cases; written into the results file first
- runner: the launch loop, independent of the platform; writes results.json and results.md after every launch
- kube, dockerapi: the two platforms (OpenShift through the Kubernetes API; one machine through the Docker API)
- probe: time to ready and time to first answer
- guidellm: the GuideLLM command line and its report
- logs: the start-up breakdown and the version from the server's own log
- energy: GPU power (DCGM through the cluster's monitoring, or nvidia-smi) and tokens per joule
- analyze: the verdicts, from the launches in a results file (the adapter boundary: rows of round, position, side,
  case and value, plus which side is whose twin, in; the analysis out)
- stats: the sign test, Holm's correction, the noise floor and the verdict rule; standard library only
- report: the short markdown table
- show, words: the comparison notebook's display and its fixed texts

Run it with `python -m rhvp_compare --help`.
"""
