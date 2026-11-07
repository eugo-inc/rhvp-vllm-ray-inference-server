# SPDX-License-Identifier: Apache-2.0
"""The sample notebook's helpers: rhvp_bench.run is what the notebook's code cells call.

- words: every sentence and label on screen
- draw: the HTML (side banners, result cards, the neutral notice, the amber strips)
- showing: the notebook's cells, in order (make_notebook.py writes them)
- measure: timing the server through its OpenAI-compatible API, and the server pod's start from Kubernetes
- replay: REPLAY mode, a stand-in server that streams a recorded run
- prompts: the seeded prompt set (prompts.jsonl)

Importing the package imports none of them, so make_notebook.py runs with the standard library alone.
"""
