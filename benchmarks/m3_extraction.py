"""M3 benchmark: does the model pick the right memory operations?

Runs the extractor on hand-written lesson exchanges whose expected
operations are known, then reports precision (operations proposed that
were expected) and recall (expected operations that were proposed), plus
how many proposed operations the validation layer would reject.

An operation matches when its type matches and, for word / profile
operations, the word or key matches too (mistake patterns are free text,
so only their type is compared).

Usage (needs a running Ollama):
  .venv/bin/python -m benchmarks.m3_extraction --model qwen3.8:4b
"""

import argparse
import json
import time

from sensai.memory.extractor import ExtractionError, MemoryExtractor
from sensai.memory.learner import LearnerMemory, Word, norm
from sensai.memory.store import MemoryStore
from sensai.ollama_client import OllamaClient

PROFILE = {"langue_cible": "japonais", "langue_maternelle": "français"}
KNOWN = [Word("inu", "chien", 1, 1, 0, 0), Word("mizu", "eau", 0, 0, 1, 0)]

# (learner message, tutor answer, expected operations as (op, key))
CASES = [
    ("Comment on dit « chat » en japonais ?",
     "On dit « neko » (猫). Exemple : neko ga suki desu = j'aime les chats.",
     [("add_word", "neko")]),
    ("Exercice : « inu » veut dire… chien !",
     "Bravo, c'est exact : inu = chien.",
     [("review_word", "inu")]),
    ("« mizu » ça veut dire feu ?",
     "Non : mizu = eau. Le feu se dit « hi ».",
     [("review_word", "mizu"), ("add_word", "hi")]),
    ("Watashi ga Camille desu.",
     "Presque ! Pour te présenter on utilise la particule « wa » : watashi wa Camille desu.",
     [("add_mistake", None)]),
    ("Je suis débutant et j'apprends pour un voyage à Osaka.",
     "Super, on va travailler les bases utiles en voyage !",
     [("set_profile", None), ("set_profile", None)]),
    ("Tu peux oublier le mot « mizu », je l'ai mal noté.",
     "D'accord, je le retire de ta liste.",
     [("delete_word", "mizu")]),
    ("Merci, à demain !",
     "À demain, bon courage !",
     []),
]


def key_of(op: dict) -> str | None:
    if op.get("op") in ("add_word", "review_word", "delete_word"):
        return norm(str(op.get("word", "")))
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.8:4b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()

    client = OllamaClient(args.host, args.model, num_ctx=4096, think=False)
    client.check()
    extractor = MemoryExtractor(client)

    tp = fp = fn = rejected = failures = 0
    rows = []
    start = time.perf_counter()
    for user_msg, answer, expected in CASES:
        try:
            ops = extractor.extract(user_msg, answer, PROFILE, KNOWN, [])
        except ExtractionError as exc:
            failures += 1
            fn += len(expected)
            rows.append({"message": user_msg, "error": str(exc)})
            print(f"✘ {user_msg}\n    extraction échouée : {exc}")
            continue

        # Would the validation layer accept these ops? Apply them to a throwaway DB.
        store = MemoryStore(":memory:")
        memory = LearnerMemory(store.conn)
        for w in KNOWN:
            memory.upsert_word("bench", "japonais", w.word, w.translation)
        report = memory.apply("bench", "japonais", ops, store.set_profile)
        rejected += len(report.rejected)
        store.close()

        remaining = list(expected)
        for op in ops:
            match = next((e for e in remaining if e[0] == op.get("op") and (e[1] is None or e[1] == key_of(op))), None)
            if match:
                remaining.remove(match)
                tp += 1
            else:
                fp += 1
        fn += len(remaining)
        ok = not remaining and len(ops) == len(expected)
        rows.append({"message": user_msg, "ops": ops, "expected": expected, "missing": remaining,
                     "rejected": report.rejected})
        print(f"{'✔' if ok else '~'} {user_msg}\n    proposé : {[(o.get('op'), key_of(o)) for o in ops]}"
              f"\n    attendu : {expected}" + (f"\n    rejeté : {report.rejected}" if report.rejected else ""))

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    elapsed = time.perf_counter() - start
    print(f"\nModèle {args.model}, {len(CASES)} échanges, {elapsed / len(CASES):.1f} s par extraction\n")
    print("| Précision | Rappel | Opérations rejetées | Extractions échouées |")
    print("|---|---|---|---|")
    print(f"| {100 * precision:.0f} % | {100 * recall:.0f} % | {rejected} | {failures}/{len(CASES)} |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
