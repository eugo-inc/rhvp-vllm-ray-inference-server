# SPDX-License-Identifier: Apache-2.0
"""The prompt set: prompts.jsonl, written from a seed by make_notebook.py and committed.

Each prompt is a run of plain words picked from VOCABULARY with a seeded generator, every prompt the same number of
words (so about the same number of tokens), every prompt different from its first word on. measure.py picks seeded
subsets of them, and puts a fresh tag in front of each prompt it sends: no two requests share a prefix, in one run or
across runs. Stdlib only.
"""

import json
import random
from pathlib import Path

SEED = 7
COUNT = 64
WORDS_PER_PROMPT = 256
WORDS_PER_SENTENCE = 16
PATH = Path(__file__).resolve().parent.parent / "prompts.jsonl"

VOCABULARY = (
    "apple", "river", "garden", "window", "pencil", "mountain", "harbor", "lantern", "meadow", "bridge",
    "orchard", "violin", "kettle", "compass", "blanket", "candle", "ladder", "basket", "feather", "island",
    "valley", "market", "library", "station", "village", "forest", "desert", "glacier", "canyon", "prairie",
    "lighthouse", "workshop", "kitchen", "balcony", "chimney", "cottage", "harvest", "festival", "journey", "letter",
    "notebook", "picture", "pocket", "ribbon", "saddle", "teapot", "tunnel", "umbrella", "wagon", "whistle",
    "anchor", "barrel", "cabinet", "carpet", "castle", "cellar", "circle", "cloud", "coffee", "copper",
    "cotton", "crystal", "curtain", "dolphin", "engine", "fabric", "falcon", "fountain", "galaxy", "granite",
    "hammer", "helmet", "honey", "jacket", "jungle", "kitten", "lemon", "marble", "mirror", "needle",
    "ocean", "olive", "orange", "paddle", "parrot", "pepper", "pillow", "planet", "pottery", "puzzle",
    "rabbit", "rocket", "salmon", "silver", "spider", "spring", "summer", "sunset", "thunder", "ticket",
    "tiger", "timber", "tomato", "turtle", "velvet", "walnut", "winter", "wizard", "yellow", "zebra",
    "quiet", "bright", "gentle", "rapid", "golden", "hollow", "humble", "narrow", "patient", "polite",
    "silent", "simple", "steady", "sudden", "tender", "vivid", "warm", "wild", "wooden", "young",
    "ancient", "brave", "calm", "clever", "curious", "distant", "eager", "fancy", "fresh", "friendly",
    "builds", "carries", "collects", "draws", "explores", "follows", "gathers", "holds", "imagines", "joins",
    "keeps", "learns", "lifts", "measures", "notices", "opens", "paints", "plants", "reads", "remembers",
    "shares", "shows", "sings", "sorts", "studies", "teaches", "travels", "visits", "watches", "writes",
    "above", "across", "after", "along", "around", "before", "behind", "below", "beside", "between",
    "beyond", "inside", "near", "over", "under", "within", "toward", "through", "the", "a",
    "every", "each", "some", "many", "few", "several", "one", "two", "three", "seven",
)


def make(seed: int = SEED, count: int = COUNT, words: int = WORDS_PER_PROMPT) -> list[str]:
    """The prompt texts the seed writes: `count` prompts of `words` words each, all different."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        picked = [rng.choice(VOCABULARY) for _ in range(words)]
        sentences = []
        for start in range(0, words, WORDS_PER_SENTENCE):
            sentence = picked[start:start + WORDS_PER_SENTENCE]
            sentences.append(" ".join([sentence[0].capitalize(), *sentence[1:]]) + ".")
        text = " ".join(sentences)
        if text not in out:
            out.append(text)
    return out


def text(prompts: list[str]) -> str:
    """prompts.jsonl's contents: one JSON object per line, {"id": n, "text": ...}."""
    return "".join(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(prompts))


def load(path: Path = PATH) -> list[str]:
    """The prompt texts in prompts.jsonl, in order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["text"] for line in lines if line.strip()]


def pick(pool: list[str], count: int, seed: int, avoid: frozenset[int] = frozenset()) -> list[int]:
    """`count` distinct prompt numbers from `pool`, chosen with `seed`, none of them in `avoid`."""
    free = [i for i in range(len(pool)) if i not in avoid]
    if count > len(free):
        raise ValueError(f"asked for {count} prompts, but only {len(free)} are left in the prompt set")
    return random.Random(seed).sample(free, count)
