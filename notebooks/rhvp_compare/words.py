# SPDX-License-Identifier: Apache-2.0
"""Every sentence and label the comparison notebook shows, as constants.

Figures come from a results file; every sentence around them is written here. No sentence gives a cause for a
result: the notebook says what was measured and how, never why one image is faster. Ratios are B/A, side A being
the baseline.
"""

TITLE = "Compare two inference-server images"
INTRO = ("This notebook shows the results of the pattern's image comparison: two server images, the same model, the "
         "same GPU and the same requests, measured in rounds that take turns between them (A B, B A, A B, ...). The "
         "comparison runs as a Job, not in this notebook; it is switched on with global.compare in "
         "values-global.yaml, and every run writes one results.json to the results volume.")
INTRO_HOW = ("Run the cells from the top. Each cell reads the newest run; pass a run id, for example "
             "show.verdicts(\"my-run-1a2b3c4d\"), to see an older one. To check the layout without a cluster, start "
             "JupyterLab with RHVP_COMPARE_REPLAY set to compare-recordings/synthetic.json.")
SECTION_HEADING = "## {number} · {title}"
SECTION_TITLE = {
    1: "Verdicts",
    2: "Start-up",
    3: "Energy per token",
    4: "What was compared, and how",
}
SECTION_TEXT = {
    1: ("One row per metric. B/A is the median, over the pairs, of side B's figure divided by side A's in the same "
        "pair; the interval next to it is a distribution-free confidence interval for that median. The verdict "
        "follows a rule fixed before the first round: TIE when the median and its whole interval lie inside the "
        "tie band; outside the band, B is FASTER or SLOWER (MORE_EFFICIENT or LESS_EFFICIENT for tokens per joule) "
        "only when the p-value is at most alpha and the test leans the same way as the median; otherwise "
        "INCONCLUSIVE. A metric is also INCONCLUSIVE when one side failed in some pair on its own account, or when "
        "the rounds ran on more than one node. Only the primary metrics, declared before the first round, count as "
        "findings: their p-values are corrected for the number of primary metrics. The secondary metrics below them "
        "use the same rule on an uncorrected p-value, so their verdicts only describe the run."),
    2: ("Time to ready runs from just before the runner asks for the side's pod until the server lists the model; "
        "time to first answer, until the first generated text of a streamed request comes back. Both are timed by "
        "the runner, the same way for every image, and the warm-up rounds (which pull the image and download the "
        "model) are left out. The phases below come from the servers' own log lines, as each server words them."),
    3: ("Output tokens per joule at each concurrency level: the level's output tokens per second divided by the mean "
        "GPU power over the level's time window. The power is GPU-only power, the board power the GPU reports; the "
        "rest of the machine is not in it."),
    4: ("The images as the node pulled them, the versions the servers reported, the GPU and the node, and the "
        "method declared before the first round. With A/A pairs in the plan, the last table shows how far two "
        "rounds of the same image land apart."),
}

# ---- strips and notices ----

REPLAY_STRIP = ("REPLAY: the results below come from a file written by a script and never measured. They only show "
                "the layout.")
SYNTHETIC_STRIP = "SYNTHETIC: this results file was written by a script and never measured. Not a result."
NO_RESULTS = "No comparison results yet."
NO_RESULTS_LINES = (
    "Switch the comparison on with global.compare.enabled: true in values-global.yaml in your fork, and push.",
    "Results appear here after the comparison's first round has finished.",
)
NOT_FOUND = "No run with the id {run} was found. Runs found: {runs}."
UNREADABLE = "A results file could not be read ({error})."
REPLAY_UNUSABLE = "REPLAY is on, but its results file could not be read ({error})."

# ---- the run line above every result ----

RUN_LINE = "Run {run} · {status} · {done} of {planned} rounds measured · {platform}"
RUN_FAILED_ROUNDS = "Rounds that failed: {rounds}."
RUN_CODE = "Why the run did not finish: {code}."
RUN_RETRIED = ("Rounds whose side was started once more after vLLM's start-up memory check failed (timed from "
               "the last start): {rounds}.")
RUN_LOAD_RETRIED = "Rounds whose load ran again after GuideLLM timed out: {rounds}. {how}"
LOAD_RETRIED_PAIRS = ("The side was taken down and started again, timed from that new start, and GuideLLM ran once "
                      "more with the same requests; every figure of the round comes from that second try.")
RETRIED_ROUND = "{index} (side {side})"
RUN_LEVELS = "Load levels left out, where requests failed or none finished: {levels}."
RUN_SERVER = "The live server was not brought back after the run; the results' run.errors say how to bring it back."
RUN_LOST = "Pairs lost to one side failing, per metric: {lost}. Those metrics are INCONCLUSIVE."
RUN_OTHERS = "Other runs in the results volume: {runs}."
SIDE_LINE = "A = {a} · B = {b}"

# ---- section 1: verdicts ----

VERDICT_HEADERS = ("Metric", "A median", "B median", "B/A median [interval]", "Pairs", "p, corrected", "Verdict")
SECONDARY_HEADERS = ("Metric", "A median", "B median", "B/A median [interval]", "Pairs", "p, not corrected",
                     "Verdict")
SECONDARY_MARK = "descriptive"
SECONDARY_NOTE = ("A verdict marked descriptive belongs to a secondary metric: its p-value is not corrected for the "
                  "number of metrics, so it describes this run and is not a finding.")
PRIMARY_TITLE = "Primary metrics"
SECONDARY_TITLE = "Secondary metrics: descriptive only, not corrected for the number of metrics"
FLOOR_NOTE = ("With {pairs} pairs and {family} primary metrics the smallest p-value the correction can give is "
              "{floor:.3g}, above alpha {alpha:g}: no primary metric can be FASTER or SLOWER in this run. {needed} "
              "pairs would be needed.")
METHOD_LINE = ("Exact {test} test on the per-pair ratios, {correction} correction over {family} primary metrics, alpha "
               "{alpha:g}, tie band {band} (B/A between {low:.3f} and {high:.3f}); all declared before the first "
               "round.")
TEST_NAMES = {"sign": "sign", "wilcoxon": "Wilcoxon signed-rank"}
CORRECTION_NAMES = {"holm": "Holm", "bonferroni": "Bonferroni", "bh": "Benjamini-Hochberg", "none": "no"}
NO_FIGURE = "-"

# ---- section 2: start-up ----

STARTUP_HEADERS = ("", "A median", "B median", "B/A median", "Verdict")
STARTUP_READY = "time to ready"
STARTUP_FIRST = "time to first answer"
PHASE_HEADERS = ("Phase, from the server's log", "A median", "B median", "Rounds (A, B)")
PHASE_NONE = "No phase of the start was found in the servers' logs."
PHASE_NOTE = ("A phase shows only when a server logs it in a form this notebook knows; a missing phase says nothing "
              "about the server.")

# ---- section 3: energy ----

ENERGY_HEADERS = ("Concurrent streams", "A tokens/J", "B tokens/J", "B/A median [interval]", "A mean power",
                  "B mean power", "Verdict")
ENERGY_OFF = "Energy was not measured in this run (energy source: none)."
ENERGY_SOURCE = {"dcgm": "DCGM, read from the cluster's monitoring", "nvidia-smi": "nvidia-smi, sampled every 500 ms"}
ENERGY_LINE = "Source: {source}. {label}."
ENERGY_MISSING = "No energy figure for: {what}. The results file says why, in each level's energy note."

# ---- section 4: provenance ----

SIDES_HEADERS = ("Side", "Name", "Image", "As pulled", "Server version")
MACHINE_LINE = "GPU: {gpu} · count {count} · driver {driver} · node type {node} · {arch} · OpenShift {openshift}"
LOAD_LINE = ("Load: GuideLLM {version}, {prompt} prompt and {output} output tokens per request, {seconds:g} s at "
             "each of {levels} concurrent streams, seed {seed}. Server seed {server_seed}.")
PLAN_LINE = "Plan: {pairs} pairs in ABBA order{warmup}{aa}."
PLAN_WARMUP = ", after one warm-up round per side"
PLAN_AA = ", then {n} A/A pairs"
NOISE_HEADERS = ("Metric", "A/A pairs", "Typical change", "Largest change")
NOISE_NOTE = "Side A against itself: how far two rounds of the same image land apart."
