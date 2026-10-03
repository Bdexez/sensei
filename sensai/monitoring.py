"""Metrics view over the JSONL event log (EV3).

    python -m sensai.monitoring                 # dashboard of the whole log
    python -m sensai.monitoring --last 200      # only the last 200 events
    python -m sensai.monitoring --json          # raw metrics, for scripts

The same summary is shown in the chat with /metrics.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Config, _load_file


def load_events(path: str | Path, last: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated line must not hide the rest
    return events[-last:] if last else events


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _latency(values: list[float]) -> dict:
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 1) if values else 0.0,
        "p95_ms": round(_percentile(values, 95), 1),
        "max_ms": round(max(values), 1) if values else 0.0,
    }


def compute_metrics(events: list[dict]) -> dict:
    model_calls = [e for e in events if e["event"] == "model_call"]
    tools = [e for e in events if e["event"] == "tool_call"]
    runs = [e for e in events if e["event"] == "react_run"]
    steps = [e for e in events if e["event"] == "react_step"]
    confirms = [e for e in events if e["event"] == "confirmation"]
    files = [e for e in events if e["event"] == "file_access"]
    sandbox = [e for e in events if e["event"] == "sandbox_run"]

    per_tool: dict[str, dict] = defaultdict(lambda: {"calls": 0, "ok": 0, "error": 0, "denied": 0, "refused": 0, "ms": []})
    for e in tools:
        t = per_tool[e.get("tool", "?")]
        t["calls"] += 1
        status = e.get("status", "ok")
        t[status if status in ("ok", "error", "denied", "refused") else "error"] += 1
        t["ms"].append(e.get("duration_ms", 0))
    tool_stats = {
        name: {**{k: v for k, v in t.items() if k != "ms"}, **_latency(t["ms"])} for name, t in sorted(per_tool.items())
    }

    errors = sum(1 for e in events if e.get("status") in ("error", "timeout"))
    timeouts = sum(1 for e in events if e.get("status") == "timeout" or e.get("timed_out"))

    days: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "ms": []})
    for e in model_calls:
        d = days[datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d")]
        d["calls"] += 1
        d["errors"] += e.get("status") != "ok"
        d["ms"].append(e.get("duration_ms", 0))

    return {
        "events": len(events),
        "interactions": len({e["interaction_id"] for e in events if e.get("interaction_id")}),
        "models": dict(Counter(e.get("model", "?") for e in model_calls)),
        "model_calls": {
            **_latency([e.get("duration_ms", 0) for e in model_calls]),
            "errors": sum(1 for e in model_calls if e.get("status") != "ok"),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) or 0 for e in model_calls),
            "output_tokens": sum(e.get("output_tokens", 0) or 0 for e in model_calls),
        },
        "react": {
            "runs": len(runs),
            "outcomes": dict(Counter(e.get("outcome", "?") for e in runs)),
            "avg_steps": round(sum(e.get("steps", 0) for e in runs) / len(runs), 1) if runs else 0.0,
            "steps": len(steps),
        },
        "tools": tool_stats,
        "confirmations": dict(Counter(e.get("decision", "?") for e in confirms)),
        "file_access": {
            "allowed": sum(1 for e in files if e.get("allowed")),
            "denied": sum(1 for e in files if not e.get("allowed")),
            "by_op": dict(Counter(f"{e.get('op')}:{'ok' if e.get('allowed') else 'denied'}" for e in files)),
        },
        "sandbox": {
            **_latency([e.get("duration_ms", 0) for e in sandbox]),
            "outcomes": dict(Counter(e.get("outcome", "?") for e in sandbox)),
        },
        "errors": errors,
        "timeouts": timeouts,
        "error_rate": round(errors / len(events), 3) if events else 0.0,
        "per_day": {
            day: {"calls": d["calls"], "errors": d["errors"], **_latency(d["ms"])} for day, d in sorted(days.items())
        },
    }


def render_metrics(m: dict) -> str:
    if not m["events"]:
        return "Aucun événement journalisé pour l'instant."
    mc = m["model_calls"]
    lines = [
        f"Événements : {m['events']} — interactions : {m['interactions']} — "
        f"erreurs : {m['errors']} (taux {m['error_rate']:.1%}), délais dépassés : {m['timeouts']}",
        f"Modèle(s) : {', '.join(f'{k} ×{v}' for k, v in m['models'].items()) or '—'}",
        f"Appels modèle : {mc['count']} (erreurs {mc['errors']}) — latence moy. {mc['avg_ms']:.0f} ms, "
        f"p95 {mc['p95_ms']:.0f} ms — tokens prompt {mc['prompt_tokens']}, sortie {mc['output_tokens']}",
    ]
    r = m["react"]
    if r["runs"]:
        outcomes = ", ".join(f"{k} {v}" for k, v in r["outcomes"].items())
        lines.append(f"Boucles ReAct : {r['runs']} ({outcomes}) — {r['avg_steps']} étapes en moyenne")
    if m["tools"]:
        lines.append("Outils :")
        for name, t in m["tools"].items():
            lines.append(
                f"  {name:<12} {t['calls']:>3} appels — ok {t['ok']}, erreur {t['error']}, "
                f"refus permission {t['denied']}, refus utilisateur {t['refused']} — moy. {t['avg_ms']:.0f} ms"
            )
    if m["confirmations"]:
        lines.append("Confirmations : " + ", ".join(f"{k} {v}" for k, v in m["confirmations"].items()))
    fa = m["file_access"]
    if fa["allowed"] or fa["denied"]:
        lines.append(f"Accès fichiers : {fa['allowed']} autorisés, {fa['denied']} refusés")
    sb = m["sandbox"]
    if sb["count"]:
        outcomes = ", ".join(f"{k} {v}" for k, v in sb["outcomes"].items())
        lines.append(f"Sandbox : {sb['count']} exécutions ({outcomes}) — moy. {sb['avg_ms']:.0f} ms, max {sb['max_ms']:.0f} ms")
    if len(m["per_day"]) > 1:
        lines.append("Évolution par jour (appels modèle) :")
        for day, d in m["per_day"].items():
            lines.append(f"  {day}  {d['calls']:>4} appels  moy. {d['avg_ms']:>6.0f} ms  p95 {d['p95_ms']:>6.0f} ms  erreurs {d['errors']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sensai.monitoring", description="Sensai metrics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log", help="JSONL log path (default: log_path from the config)")
    parser.add_argument("--last", type=int, help="only the last N events")
    parser.add_argument("--json", action="store_true", help="print raw metrics as JSON")
    args = parser.parse_args(argv)
    path = args.log or _load_file(args.config).get("log_path", Config.log_path)
    metrics = compute_metrics(load_events(path, args.last))
    print(json.dumps(metrics, indent=2, ensure_ascii=False) if args.json else render_metrics(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
