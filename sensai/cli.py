"""Interactive CLI loop."""

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from .agent.approval import ApprovalGate, ConsoleApprover
from .agent.permissions import FileGuard
from .agent.react import AgentResult, ReActAgent, Step
from .agent.sandbox import Sandbox
from .agent.tools import ToolRegistry
from .config import Config, ConfigError, load_config
from .context import ContextBuilder
from .memory.compressor import CompressionError, Compressor, Summary, compress
from .memory.extractor import ExtractionError, MemoryExtractor
from .memory.learner import LearnerMemory
from .memory.store import MemoryStore, Message
from .memory.tokens import estimate_message
from .monitoring import compute_metrics, load_events, render_metrics
from .observability import EventLogger, new_interaction_id
from .ollama_client import ChatStats, ModelNotFound, OllamaClient, OllamaError, OllamaUnavailable

HELP = """Commandes :
  /help                      cette aide
  /new                       nouvelle session
  /sessions                  lister les sessions
  /load <id>                 reprendre une session
  /profile                   afficher le profil
  /profile set <clé> <val>   définir une info (ex. /profile set langue_cible japonais)
  /profile del <clé>         supprimer une info
  /context                   détail du dernier contexte envoyé
  /summary                   afficher le résumé des anciens échanges
  /compress                  forcer la compression des anciens échanges
  /vocab                     vocabulaire appris (révision espacée)
  /vocab del <mot>           oublier un mot
  /mistakes                  erreurs récurrentes
  /memory                    bloc mémoire injecté au dernier message
  /agent <tâche>             mode agent : l'agent lit/écrit dans le workspace et exécute du code
                             en sandbox, étape par étape (confirmation avant toute modification)
  /trace                     étapes de la dernière tâche agent
  /tools                     outils de l'agent, permissions et limites
  /metrics                   métriques du système (latence, tokens, erreurs, outils…)
  /quit                      quitter"""


class Chat:
    def __init__(
        self,
        config: Config,
        store: MemoryStore,
        client: OllamaClient,
        logger: EventLogger | None = None,
        agent: ReActAgent | None = None,
    ):
        self.config = config
        self.logger = logger or client.logger
        self.agent = agent
        self.last_agent: AgentResult | None = None
        self.store = store
        self.client = client
        self.context = ContextBuilder(
            config.system_prompt,
            config.num_ctx,
            config.response_reserve,
            summary_share=config.summary_share,
            memory_share=config.memory_share,
        )
        self.compressor = Compressor(client, config.compress_trigger, config.compress_keep)
        self.session_id = store.create_session(config.user)
        self.history: list[Message] = []  # unsummarized turns only
        self.summary: Summary | None = None
        self.compress_backoff = 0  # turns to wait after a failed compression
        self.learner = LearnerMemory(store.conn)
        self.extractor = MemoryExtractor(client)
        # Extraction runs in the background while the learner reads the answer;
        # only the model call happens there, results are applied on this thread.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None
        self.last_memory: str | None = None
        self.last_report = None

    @property
    def language(self) -> str:
        return self.store.get_profile(self.config.user).get("langue_cible", "")

    def close(self) -> None:
        self._apply_pending()
        self.executor.shutdown()

    def handle_command(self, line: str) -> bool:
        """Return False to quit."""
        parts = line.split(maxsplit=3)
        cmd = parts[0]
        user = self.config.user
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/new":
            self.session_id = self.store.create_session(user)
            self.history = []
            self.summary = None
            print(f"Nouvelle session #{self.session_id}")
        elif cmd == "/sessions":
            for s in self.store.list_sessions(user):
                when = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
                mark = "*" if s.id == self.session_id else " "
                print(f"{mark} #{s.id:<4} {when}  {s.message_count:>3} msg  {s.title or '(vide)'}")
        elif cmd == "/load":
            if len(parts) < 2 or not parts[1].isdigit():
                print("Usage : /load <id>")
            elif not self.store.session_exists(user, int(parts[1])):
                print(f"Session #{parts[1]} introuvable")
            else:
                self.session_id = int(parts[1])
                saved = self.store.get_summary(self.session_id)
                self.summary = Summary.from_json(saved[0]) if saved else None
                self.history = self.store.load_messages(self.session_id, after_id=saved[1] if saved else 0)
                total = self.store.count_messages(self.session_id)
                print(
                    f"Session #{self.session_id} rechargée ({total} messages, "
                    f"dont {total - len(self.history)} résumés)"
                )
        elif cmd == "/profile":
            self._profile(parts[1:])
        elif cmd == "/context":
            self._print_report()
        elif cmd == "/summary":
            print(self.summary.render() if self.summary else "Aucun résumé pour cette session.")
        elif cmd == "/compress":
            if not self._compress(force=True):
                print("Rien à compresser (il faut au moins 2 messages en dehors des plus récents).")
        elif cmd == "/vocab":
            self._apply_pending()
            if len(parts) >= 3 and parts[1] == "del":
                word = line.split(maxsplit=2)[2]
                print("Oublié" if self.learner.delete_word(user, self.language, word) else "Mot absent")
            else:
                words = self.learner.words(user, self.language)
                if not words:
                    print("Aucun mot enregistré pour l'instant.")
                for w in words:
                    when = datetime.fromtimestamp(w.next_review).strftime("%Y-%m-%d")
                    print(f"  [boîte {w.box}] {w.line()}  — révision {when}")
        elif cmd == "/mistakes":
            self._apply_pending()
            mistakes = self.learner.mistakes(user, self.language, limit=50)
            if not mistakes:
                print("Aucune erreur enregistrée.")
            for m in mistakes:
                print(f"  {m.line()}")
        elif cmd == "/memory":
            print(self.last_memory or "Aucun bloc mémoire injecté au dernier message.")
        elif cmd == "/agent":
            task = line[len("/agent"):].strip()
            if not task:
                print("Usage : /agent <tâche>   ex. /agent crée une fiche de révision à partir de docs/lecon.md")
            else:
                self.run_agent(task)
        elif cmd == "/trace":
            print(self.last_agent.trace() if self.last_agent else "Aucune tâche agent pour l'instant.")
        elif cmd == "/tools":
            if self.agent is None:
                print("Mode agent indisponible.")
            else:
                print(self.agent.tools.describe())
                print(self.agent.tools.guard.describe())
                print(self.agent.tools.sandbox.describe())
                levels = ", ".join(sorted(self.agent.tools.gate.levels))
                print(f"Confirmation demandée pour : {levels}. Étapes max : {self.agent.max_steps}.")
        elif cmd == "/metrics":
            print(render_metrics(compute_metrics(load_events(self.config.log_path))))
        else:
            print(f"Commande inconnue : {cmd} (voir /help)")
        return True

    def _profile(self, args: list[str]) -> None:
        user = self.config.user
        if not args:
            profile = self.store.get_profile(user)
            if not profile:
                print("Profil vide. Ex. : /profile set langue_cible espagnol")
            for k, v in profile.items():
                print(f"  {k}: {v}")
        elif args[0] == "set" and len(args) == 3:
            self.store.set_profile(user, args[1], args[2])
            print(f"{args[1]} = {args[2]}")
        elif args[0] == "del" and len(args) == 2:
            print("Supprimé" if self.store.delete_profile_key(user, args[1]) else "Clé absente")
        else:
            print("Usage : /profile | /profile set <clé> <valeur> | /profile del <clé>")

    def _print_report(self) -> None:
        r = self.last_report
        if r is None:
            print("Aucun message envoyé pour l'instant.")
            return
        print(
            f"Budget {r.used}/{r.budget} tokens (estimés) — système+profil {r.system_tokens}, "
            f"résumé {r.summary_tokens}/{self.context.summary_cap}, RAG {r.rag_tokens}, "
            f"historique {r.history_tokens}/{self.context.history_budget(self.store.get_profile(self.config.user))} "
            f"({r.history_kept} messages gardés, {r.history_dropped} écartés)"
        )
        print(f"Mémoire de l'apprenant {r.memory_tokens}/{self.context.memory_cap}")
        o = self.learner.op_stats(self.config.user)
        if o["total"]:
            print(
                f"Opérations mémoire : {o['applied']}/{o['total']} appliquées, "
                f"{o['failed_extractions']} extractions échouées"
            )
        s = self.store.compression_stats(self.session_id)
        if s["passes"]:
            print(
                f"Compressions : {s['ok']}/{s['passes']} réussies, {s['messages']} messages résumés, "
                f"~{s['saved']} tokens économisés, {s['avg_ms']:.0f} ms en moyenne"
            )

    def _compress(self, force: bool = False) -> bool:
        """Summarize old turns if needed; return True if something was compressed."""
        available = self.context.history_budget(self.store.get_profile(self.config.user))
        done = False
        while chunk := self.compressor.select(self.history, available, force=force and not done):
            print(f"[compression de {len(chunk)} anciens messages…]", file=sys.stderr)
            try:
                result = compress(self.compressor, chunk, self.summary, self.context.summary_cap)
            except CompressionError as exc:
                self.store.log_compression(self.session_id, len(chunk), 0, 0, 0, error=str(exc))
                self.compress_backoff = 3
                print(f"[compression échouée : {exc} — les plus vieux messages seront écartés]", file=sys.stderr)
                return done
            self.summary = result.summary
            self.store.set_summary(self.session_id, result.summary.to_json(), result.covered_until, result.tokens_after)
            self.store.log_compression(
                self.session_id, result.messages_in, result.tokens_before, result.tokens_after, result.duration_ms
            )
            self.history = self.history[len(chunk):]
            done = True
            print(
                f"[{result.tokens_before} → {result.tokens_after} tokens en {result.duration_ms / 1000:.1f} s"
                + (f", {result.dropped_items} éléments anciens retirés du résumé" if result.dropped_items else "")
                + "]",
                file=sys.stderr,
            )
        return done

    def _apply_pending(self) -> None:
        """Wait for the background extraction of the previous exchange and apply it."""
        if self.pending is None:
            return
        future, self.pending = self.pending, None
        user = self.config.user
        try:
            ops = future.result()
        except ExtractionError as exc:
            self.learner.log_failure(user, self.session_id, str(exc))
            print(f"[mémoire : extraction échouée ({exc})]", file=sys.stderr)
            return
        if not ops:
            return
        report = self.learner.apply(user, self.language, ops, self.store.set_profile, self.session_id)
        print(f"[mémoire : {report.short()}]", file=sys.stderr)

    def _start_extraction(self, user_text: str, answer: str) -> None:
        user = self.config.user
        language = self.language
        profile = self.store.get_profile(user)
        exchange = f"{user_text}\n{answer}"
        known_words = self.learner.mentioned_words(user, language, exchange, limit=20)
        known_mistakes = self.learner.mistakes(user, language, limit=10)
        self.pending = self.executor.submit(
            self.extractor.extract, user_text, answer, profile, known_words, known_mistakes
        )

    def run_agent(self, task: str) -> AgentResult | None:
        """A1: run the ReAct loop on `task`; the exchange is kept in the chat history."""
        if self.agent is None:
            print("Mode agent indisponible.")
            return None
        self._apply_pending()
        profile = self.store.get_profile(self.config.user)
        context = ("Profil de l'apprenant : " + ", ".join(f"{k} = {v}" for k, v in profile.items())) if profile else None
        print(f"[agent — {self.agent.max_steps} étapes max, Ctrl-C pour interrompre]")
        try:
            result = self.agent.run(task, context)
        except KeyboardInterrupt:
            print("\n[agent interrompu]")
            return None
        self.last_agent = result
        label = {"done": "terminé", "max_steps": "limite d'étapes atteinte", "error": "arrêté sur erreur"}[result.outcome]
        print(f"\n[agent {label} — {len(result.steps)} étapes, {result.duration_ms / 1000:.1f} s, /trace pour le détail]")
        print(result.answer)
        for role, content in (("user", f"[tâche agent] {task}"), ("assistant", result.answer)):
            msg = Message(role, content, estimate_message(content))
            self.history.append(msg)
            self.store.add_message(self.session_id, msg)
        return result

    def send(self, text: str) -> None:
        interaction_id = new_interaction_id()
        with self.logger.timed("chat_turn", interaction_id, session_id=self.session_id, input_chars=len(text)) as ev:
            self._send(text, interaction_id, ev)

    def _send(self, text: str, interaction_id: str, ev: dict) -> None:
        self._apply_pending()
        user_msg = Message("user", text, estimate_message(text))
        self.history.append(user_msg)
        self.store.add_message(self.session_id, user_msg)

        if self.config.compression:
            if self.compress_backoff:
                self.compress_backoff -= 1
            else:
                self._compress()
        summary = self.summary.render() if self.summary and not self.summary.is_empty() else None
        self.last_memory = (
            self.learner.render(self.config.user, self.language, text, self.context.memory_cap)
            if self.config.memory
            else None
        )
        messages, self.last_report = self.context.build(
            self.store.get_profile(self.config.user), self.history, summary, self.last_memory
        )
        if self.last_report.history_dropped:
            print(f"[{self.last_report.history_dropped} anciens messages hors contexte]", file=sys.stderr)

        stats = ChatStats()
        parts: list[str] = []
        try:
            for piece in self.client.chat_stream(messages, stats, interaction_id=interaction_id):
                print(piece, end="", flush=True)
                parts.append(piece)
        except KeyboardInterrupt:
            print("\n[génération interrompue]")
            ev["status"] = "interrupted"
        finally:
            print()
        answer = "".join(parts)
        ev.update(
            context_tokens=self.last_report.used,
            history_dropped=self.last_report.history_dropped,
            output_chars=len(answer),
        )
        if answer:
            msg = Message("assistant", answer, stats.eval_count or estimate_message(answer))
            self.history.append(msg)
            self.store.add_message(self.session_id, msg)
            if self.config.memory:
                self._start_extraction(text, answer)


def print_step(step: Step) -> None:
    print(step.line(), flush=True)
    error = step.observation.get("error")
    if error and step.status != "ok":
        print(f"           {error}")
    elif "outcome" in step.observation:
        out = (step.observation.get("stdout") or step.observation.get("stderr") or "").strip().splitlines()
        tail = f" — {out[-1][:100]}" if out else ""
        print(f"           sandbox : {step.observation['outcome']}, code {step.observation.get('exit_code')}{tail}")


def build_agent(config: Config, client: OllamaClient, logger: EventLogger, budget: int) -> ReActAgent:
    """Wire the ReAct loop to its tools: permissions (T4), sandbox (T2), confirmation (A3), logs (EV3)."""
    guard = FileGuard(
        config.file_roots,
        allowed_ext=config.file_allowed_ext,
        denied_ext=config.file_denied_ext,
        max_bytes=config.file_max_bytes,
        allow_symlinks=config.file_allow_symlinks,
        logger=logger,
    )
    sandbox = Sandbox(
        runtimes=config.sandbox_runtimes,
        timeout=config.sandbox_timeout,
        max_output=config.sandbox_max_output,
        memory_mb=config.sandbox_memory_mb,
        backend=config.sandbox_backend,
        docker_image=config.sandbox_docker_image,
        logger=logger,
    )
    gate = ApprovalGate(ConsoleApprover(timeout=config.confirm_timeout), config.confirm_actions, logger)
    return ReActAgent(
        client,
        ToolRegistry(guard, sandbox, gate, logger),
        logger,
        max_steps=config.agent_max_steps,
        max_parse_errors=config.agent_max_parse_errors,
        observation_chars=config.agent_observation_chars,
        token_budget=budget,
        on_step=print_step,
    )


def run(config: Config) -> int:
    logger = EventLogger(config.log_path, config.log_content_chars, config.log_max_bytes)
    client = OllamaClient(config.host, config.model, config.num_ctx, config.think, logger=logger)
    try:
        client.check()
    except OllamaError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    store = MemoryStore(config.db_path)
    chat = Chat(config, store, client, logger)
    try:
        chat.agent = build_agent(config, client, logger, chat.context.budget)
    except (ValueError, OSError) as exc:
        print(f"Mode agent désactivé : {exc}", file=sys.stderr)
    profile = store.get_profile(config.user)
    print(f"Sensai — modèle {config.model}, apprenant « {config.user} », session #{chat.session_id}")
    if not profile:
        print("Astuce : renseigne ton profil, ex. /profile set langue_cible japonais")
    print("/help pour les commandes.\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not chat.handle_command(line):
                    break
                continue
            try:
                chat.send(line)
            except ModelNotFound as exc:
                print(f"Erreur : {exc}", file=sys.stderr)
            except OllamaUnavailable as exc:
                print(f"Erreur : {exc} — réessaie quand Ollama est relancé.", file=sys.stderr)
            except OllamaError as exc:
                print(f"Erreur du modèle : {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print()
    finally:
        chat.close()
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    return run(config)
