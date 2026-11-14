# SPDX-License-Identifier: Apache-2.0
"""Every sentence and label the comparison notebook shows, as constants.

Figures come from a results file; every sentence around them is written here. No sentence gives a cause for a
result: the notebook says what was measured and how, never why one image is faster. Ratios are B/A, side A being
the baseline.
"""

TITLE = "Compare two inference-server images"
INTRO = ("This notebook shows the results of the pattern's image comparison: two server images, the same model, the "
         "same GPU and the same requests. Each side also runs against an identical copy of itself, its twin, in the "
         "same rounds, which measures how far two launches of one image land apart. A round launches A, A~same, B "
         "and B~same once each, in an order that rotates from round to round. The comparison runs as a Job, not in "
         "this notebook; it is switched on with global.compare in values-global.yaml, and every run writes one "
         "results.json to the results volume.")
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
    1: ("One row per case. B/A is the median, over the rounds, of side B's figure divided by side A's in the same "
        "round. The bar is the noise the twins measured: side B's noise floor plus side A's, where a side's floor is "
        "how far the median of side / twin lies from 1, the largest over the cases of one metric and role (primary "
        "or secondary). The verdict "
        "follows a rule fixed before the first round, from the exact sign test over the rounds: outside the bar and "
        "significant, B is FASTER or SLOWER (times) or BETTER or WORSE (rates and tokens per joule); inside the bar "
        "and not significant, TIE; inside the bar but significant, or outside the bar but not significant, "
        "UNRESOLVED. A case that cannot be graded has NO VERDICT, with its reason: too few rounds, a missing or zero "
        "floor, a figure of 0, a side that failed on its own account in some round, or launches on more than one "
        "node. A win, a loss and a tie need the same evidence. Only the primary cases, declared before the first "
        "round, count as findings: their p-values are corrected for testing them all at once. The secondary cases "
        "below them use the same rule on an uncorrected p-value, so their verdicts only describe the run."),
    2: ("Time to ready runs from just before the runner asks for the side's pod until the server lists the model; "
        "time to first answer, until the first generated text of a streamed request comes back. Both are timed by "
        "the runner, the same way for every image, and the warm-up launches (which pull the image and download the "
        "model) are left out. The phases below come from the servers' own log lines, as each server words them."),
    3: ("Output tokens per joule at each concurrency level: the level's output tokens per second divided by the mean "
        "GPU power over the level's time window. The power is GPU-only power, the board power the GPU reports; the "
        "rest of the machine is not in it. No figure is given from fewer power samples than the plan asks for."),
    4: ("The images as the node pulled them, the versions the servers reported, the GPU and the node, the method "
        "declared before the first round, and each side against its twin: the median of side / twin over the "
        "rounds and the noise floor it gives."),
}

# ---- strips and notices ----

REPLAY_STRIP = ("REPLAY: the results below come from a file written by a script and never measured. They only show "
                "the layout.")
SYNTHETIC_STRIP = "SYNTHETIC: this results file was written by a script and never measured. Not a result."
NO_RESULTS = "No comparison results yet."
NO_RESULTS_LINES = (
    "Switch the comparison on with global.compare.enabled: true in values-global.yaml in your fork, and push.",
    "Results appear here after the comparison's first launch has finished.",
)
NOT_FOUND = "No run with the id {run} was found. Runs found: {runs}."
UNREADABLE = "A results file could not be read ({error})."
REPLAY_UNUSABLE = "REPLAY is on, but its results file could not be read ({error})."
OLD_SCHEMA = ("This results file was written under an earlier rule ({schema}) and is not shown. Run python -m "
              "rhvp_compare analyze only on files of this version.")

# ---- the run line above every result ----

RUN_LINE = ("Run {run} · {status} · {rounds} of {planned} rounds complete · {done} of {launches} launches measured · "
            "{platform}")
RUN_FAILED_LAUNCHES = "Launches that failed: {launches}."
RUN_CODE = "Why the run did not finish: {code}."
RUN_RETRIED = ("Launches whose side was started again after vLLM's start-up memory check failed (timed from the "
               "last start), with how many times it was started in all: {launches}.")
RETRIED_LAUNCH = "{index} ({side}, {starts})"
RUN_LOAD_RETRIED = "Launches whose GuideLLM timed out: {launches}. {how}"
LOAD_RETRIED_HOW = ("After a time-out the side was taken down and the launch ran again from a new start, timed from "
                    "that start, with the same requests; every figure of such a launch comes from that second try. A "
                    "second time-out fails the launch (LOAD_FAILED).")
LOAD_RETRIED_RAN = "{index} ({side}): ran again"
LOAD_RETRIED_NOT = "{index} ({side}): did not run again ({code})"
RUN_LEVELS = "Load levels left out, where requests failed or none finished: {levels}."
RUN_LEVEL = "launch {index} ({side}) at {concurrency} streams"
ENERGY_MIXED = ("Tokens per joule did not come from one method in every launch: {levels}. The energy counter needs as "
                "many distinct readings in a level as energy.minSamples; otherwise the power samples give the mean.")
ENERGY_MIXED_LEVEL = "at {concurrency} streams, the energy counter in {counter} and the power samples in {power}"
RUN_SERVER = "The live server was not brought back after the run; the results' run.errors say how to bring it back."
RUN_LOST = ("Rounds lost to one side failing on its own account, per case: {lost}. Those cases have NO VERDICT "
            "(side_failed).")
RUN_OTHERS = "Other runs in the results volume: {runs}."
TWIN_DIFFERS = ("A side leaned the same way against its twin in more rounds than chance would usually give: {cases}. "
                "These p-values are not corrected for the {tested} comparisons of a side with its twin in this run; "
                "with that many, one this low can turn up by chance.")
TWIN_DIFFERS_CASE = "side {side} on {case} (p = {p})"
TWIN_ZERO = ("A figure of 0 in a side's comparison with its twin gives no ratio: that side has no noise floor there, "
             "and every case whose floor uses it has NO VERDICT (a figure of 0): {cases}.")
TWIN_ZERO_CASE = "{who} on {case} in {rounds} (no verdict for {affected})"
ROUND_ONE = "round {n}"
ROUND_MANY = "rounds {n}"
TWIN_ZERO_SIDE = "side {side}"
TWIN_ZERO_TWIN = "side {side}'s twin"
SIDE_LINE = "A = {a} · B = {b}"

# ---- section 1: verdicts ----

VERDICT_HEADERS = ("Case", "A median", "B median", "B/A median", "Bar (noise)", "Rounds", "p, corrected", "Verdict")
SECONDARY_HEADERS = ("Case", "A median", "B median", "B/A median", "Bar (noise)", "Rounds", "p, not corrected",
                     "Verdict")
SECONDARY_MARK = "descriptive"
SECONDARY_NOTE = ("A verdict marked descriptive belongs to a secondary case: its p-value is not corrected for the "
                  "number of cases, so it describes this run and is not a finding.")
PRIMARY_TITLE = "Primary cases ({plain})"
SECONDARY_TITLE = "Secondary cases: descriptive only, not corrected for the number of cases"
REASON_TEXT = {
    "inside_noise": "inside the bar, yet significant",
    "not_significant": "outside the bar, not significant",
    "order_unbalanced": "one side ran first far more often",
    "single_measurement": "one measurement, no p-value",
    "too_few_rounds": "too few rounds for any verdict",
    "no_floor": "no twin, no noise floor",
    "one_sided_floor": "only one side has a twin",
    "zero_floor": "a noise floor of exactly 0",
    "nonpositive_value": "a figure of 0",
    "order_not_recorded": "run order not recorded",
    "side_failed": "a side failed on its own account",
    "more_than_one_node": "launches on more than one node",
}
POWER_NOTE = ("With {rounds} rounds and {family} primary cases, a verdict on a primary case needs {agree} of the "
              "{rounds} rounds to lean the same way, and at most {most} primary cases can get one.")
POWER_NONE = ("With {rounds} rounds and {family} primary cases no primary case can get a verdict: at least {needed} "
              "rounds are needed.")
TWINS_NONE = "Without a twin for both sides no case can get a verdict."
LOAD_TIMEOUTS = ("GuideLLM timed out {a} times in the launches of side A and {b} times in those of side B, twins "
                 "included. Each such launch ran again from a new start, or failed.")
LOAD_TIMEOUTS_ONE_SIDED = ("Only side {side} timed out: a load that stalls on one side only is a difference the "
                           "verdicts above do not show.")
METHOD_LINE = ("Exact two-sided sign test over the rounds, Holm's correction over the primary cases that have a "
               "p-value ({plain}; {family} declared), alpha {alpha:g}; the bar comes from the twins. All declared "
               "before the first round.")
NO_FIGURE = "-"

# ---- section 2: start-up ----

STARTUP_HEADERS = ("", "A median", "B median", "B/A median", "Verdict")
STARTUP_READY = "time to ready"
STARTUP_FIRST = "time to first answer"
PHASE_HEADERS = ("Phase, from the server's log", "A median", "B median", "Launches (A, B)")
PHASE_NONE = "No phase of the start was found in the servers' logs."
PHASE_NOTE = ("A phase shows only when a server logs it in a form this notebook knows; a missing phase says nothing "
              "about the server.")

# ---- section 3: energy ----

ENERGY_HEADERS = ("Concurrent streams", "A tokens/J, GPU-only power", "B tokens/J, GPU-only power", "B/A median",
                  "A mean power", "B mean power", "Verdict")
ENERGY_OFF = "Energy was not measured in this run (energy source: none)."
ENERGY_SOURCE = {"dcgm": "DCGM, read from the cluster's monitoring", "nvidia-smi": "nvidia-smi, sampled every 500 ms"}
ENERGY_LINE = "Source: {source}. {label}. At least {samples} power samples per load level."
ENERGY_MISSING = "No energy figure for: {what}. The results file says why, in each level's energy note."

# ---- section 4: provenance ----

SIDES_HEADERS = ("Side", "Name", "Image", "As pulled", "Server version")
MACHINE_LINE = "GPU: {gpu} · count {count} · driver {driver} · node type {node} · {arch} · OpenShift {openshift}"
LOAD_LINE = ("Load: GuideLLM {version}, {prompt} prompt and {output} output tokens per request, {seconds:g} s at "
             "each of {levels} concurrent streams, seed {seed}. Server seed {server_seed}.")
PLAN_LINE = "Plan: {rounds} rounds of {lineup}, the order rotating from round to round{warmup}; {launches} launches."
PLAN_WARMUP = ", after one warm-up launch per side"
TWIN_HEADERS = ("Side", "Case", "Side / twin, median", "Noise floor", "Rounds", "p, not corrected")
TWIN_NOTE = ("Each side against its twin, an identical copy of it in the same rounds: how far two launches of one "
             "image land apart. A p-value below 0.05 here means the side differed from its own copy more often than "
             "chance.")
