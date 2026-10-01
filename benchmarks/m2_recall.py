"""M2 benchmark: does compression preserve facts that truncation loses?

Plants facts at the start of a long synthetic lesson, pads it with filler
turns, then asks one question per fact under three strategies:

  full      large context, nothing removed (upper bound)
  truncate  small budget, oldest turns dropped (no M2)
  compress  small budget, oldest turns summarized (M2)

A fact counts as recalled if the answer contains one of its expected keywords.

Usage (needs a running Ollama):
  .venv/bin/python -m benchmarks.m2_recall --model qwen3.8:4b --num-ctx 2048 --filler 30
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

from sensai.context import ContextBuilder
from sensai.memory.compressor import Compressor, Summary, compress
from sensai.memory.store import Message
from sensai.ollama_client import ChatStats, OllamaClient

SYSTEM = "Tu es Sensai, un tuteur de japonais. Réponds brièvement, en français."

# (what the learner says early on, question asked at the end, accepted keywords)
FACTS = [
    ("Je m'appelle Camille et j'apprends le japonais pour un voyage à Kyoto en avril.",
     "Dans quelle ville est-ce que je pars en voyage ?", ["kyoto"]),
    ("J'ai retenu que « inu » veut dire chien.",
     "Que veut dire « inu » ?", ["chien"]),
    ("Je confonds toujours les particules « wa » et « ga ».",
     "Quelles particules est-ce que je confonds souvent ?", ["ga"]),
    ("Je préfère les exercices de QCM aux traductions.",
     "Quel type d'exercice est-ce que je préfère ?", ["qcm", "choix multiple"]),
    ("Mon chat s'appelle Mochi, j'aimerais savoir le dire en japonais.",
     "Comment s'appelle mon chat ?", ["mochi"]),
    ("J'ai raté l'exercice sur les compteurs, « hon » pour les objets longs.",
     "Quel compteur ai-je raté ?", ["hon"]),
]

FILLER_USER = "Donne-moi un nouvel exercice de vocabulaire sur le thème {n}."
FILLER_ASSISTANT = (
    "Voici l'exercice {n} : traduis la phrase suivante en japonais et explique ton choix de particule. "
    "Prends ton temps, puis donne ta réponse et je la corrigerai en détaillant la règle de grammaire."
)


@dataclass
class Result:
    strategy: str
    recalled: int
    total: int
    avg_prompt_tokens: float
    compressions: int
    compression_ms: float
    details: list[dict]


def build_lesson(filler: int) -> list[Message]:
    msgs: list[Message] = []
    for statement, _, _ in FACTS:
        msgs.append(Message("user", statement))
        msgs.append(Message("assistant", "C'est noté !"))
    for n in range(1, filler + 1):
        msgs.append(Message("user", FILLER_USER.format(n=n)))
        msgs.append(Message("assistant", FILLER_ASSISTANT.format(n=n)))
    for i, m in enumerate(msgs, 1):
        m.id = i
    return msgs


def replay(lesson: list[Message], builder: ContextBuilder, compressor: Compressor | None):
    """Feed the lesson turn by turn, compressing like the CLI does."""
    history: list[Message] = []
    summary: Summary | None = None
    passes, total_ms = 0, 0.0
    for msg in lesson:
        history.append(msg)
        if compressor is None or msg.role != "user":
            continue
        available = builder.history_budget({})
        while chunk := compressor.select(history, available):
            result = compress(compressor, chunk, summary, builder.summary_cap)
            summary, history = result.summary, history[len(chunk):]
            passes += 1
            total_ms += result.duration_ms
            print(f"  compression {passes}: {result.tokens_before} → {result.tokens_after} tokens")
    return history, summary, passes, total_ms


def run(strategy: str, client: OllamaClient, lesson: list[Message], num_ctx: int, args) -> Result:
    print(f"[{strategy}]")
    builder = ContextBuilder(SYSTEM, num_ctx, args.reserve, summary_share=args.summary_share)
    compressor = Compressor(client, args.trigger, args.keep) if strategy == "compress" else None
    history, summary, passes, ms = replay(lesson, builder, compressor)

    details, recalled, prompt_tokens = [], 0, []
    for i, (_, question, keywords) in enumerate(FACTS):
        q = Message("user", question, id=10_000 + i)
        rendered = summary.render() if summary and not summary.is_empty() else None
        messages, report = builder.build({}, history + [q], rendered)
        stats = ChatStats()
        answer = client.chat(messages, stats, options={"temperature": 0})
        ok = any(k in answer.lower() for k in keywords)
        recalled += ok
        prompt_tokens.append(stats.prompt_eval_count or report.used)
        details.append({"question": question, "answer": answer.strip()[:200], "ok": ok,
                        "dropped": report.history_dropped})
        print(f"  {'✔' if ok else '✘'} {question}")
    return Result(strategy, recalled, len(FACTS), sum(prompt_tokens) / len(prompt_tokens), passes, ms, details)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.8:4b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--num-ctx", type=int, default=2048, help="small window for truncate/compress")
    p.add_argument("--full-ctx", type=int, default=32768, help="window for the 'full' upper bound")
    p.add_argument("--filler", type=int, default=30, help="filler exchanges after the facts")
    p.add_argument("--reserve", type=float, default=0.2)
    p.add_argument("--trigger", type=float, default=0.8)
    p.add_argument("--keep", type=float, default=0.4)
    p.add_argument("--summary-share", type=float, default=0.1)
    p.add_argument("--strategies", default="full,truncate,compress")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()

    lesson = build_lesson(args.filler)
    results = []
    for strategy in args.strategies.split(","):
        num_ctx = args.full_ctx if strategy == "full" else args.num_ctx
        client = OllamaClient(args.host, args.model, num_ctx, think=False)
        client.check()
        start = time.perf_counter()
        r = run(strategy, client, lesson, num_ctx, args)
        print(f"  {r.recalled}/{r.total} en {time.perf_counter() - start:.1f} s\n")
        results.append(r)

    print(f"Modèle {args.model}, {len(lesson)} messages, num_ctx {args.num_ctx} (full : {args.full_ctx})\n")
    print("| Stratégie | Rappel | Tokens de prompt (moy.) | Compressions | Temps de compression |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r.strategy} | {r.recalled}/{r.total} ({100 * r.recalled / r.total:.0f} %) | "
              f"{r.avg_prompt_tokens:.0f} | {r.compressions} | {r.compression_ms / 1000:.1f} s |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
