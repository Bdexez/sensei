# Mémoire & Contexte — fonctionnement

Ce document explique comment Sensai gère la **mémoire** (ce qu'il retient entre les messages et entre les sessions) et le **contexte** (ce qu'on envoie réellement au modèle à chaque tour).

Cas d'usage fil rouge : un **tuteur de langues façon Duolingo**, sur un modèle local Ollama (par défaut `qwen3.8:4b`, tag à vérifier avec `ollama list`).

Features du catalogue concernées :

| ID | Feature | État |
|---|---|---|
| M1 | Session Persistence & Profiles | ✅ fait |
| M2 | Token Budgeting & Semantic Compression | ✅ fait (métriques à mesurer avec le vrai modèle) |
| M3 | External Structured Memory | ⏳ à faire |
| M4 | Artifact & State Document | ❌ pas prévu pour l'instant |

---

## 1. Le problème : une seule fenêtre de contexte pour tout le monde

Un LLM ne « se souvient » de rien : à chaque tour, on lui renvoie **tout** ce qu'il doit savoir dans une seule requête. Cette requête est limitée par la taille de la fenêtre de contexte, `num_ctx` (en tokens).

Tout se partage cette fenêtre :

- le prompt système (le rôle de tuteur) ;
- le profil de l'apprenant ;
- les documents du RAG (géré par un coéquipier) ;
- l'historique de la conversation ;
- **et la réponse du modèle**, qui est générée dans la même fenêtre.

### ⚠️ Piège Ollama : la troncature silencieuse

Si la requête dépasse `num_ctx`, **Ollama supprime les plus vieux tokens sans renvoyer d'erreur**. Le modèle peut alors perdre le prompt système ou le profil sans qu'on le sache. De plus, la valeur par défaut de `num_ctx` dans Ollama est petite (2048 à 4096 selon la version).

C'est pour ça que :

1. `num_ctx` est **toujours fixé explicitement** dans chaque requête (`options.num_ctx`, valeur dans `config.json`) ;
2. c'est **notre code** qui choisit ce qui entre dans le contexte, avant l'envoi, au lieu de laisser Ollama couper au hasard.

---

## 2. Architecture

```
sensai/
  config.py           # lecture de config.json + arguments CLI
  ollama_client.py    # appels HTTP à Ollama (streaming, erreurs, compteurs de tokens)
  context.py          # ContextBuilder : assemble le prompt dans le budget
  cli.py              # boucle de chat et commandes
  memory/
    store.py          # SQLite : sessions, messages, profil (M1), résumés (M2) — accueillera M3
    tokens.py         # estimation du nombre de tokens
    compressor.py     # compression des vieux tours en résumé structuré (M2)
benchmarks/
  m2_recall.py        # banc de test : rappel des faits avec / sans compression
```

Déroulement d'un tour :

```
saisie utilisateur
   │
   ├─► sauvegarde du message en base (store.add_message)
   │
   ├─► compression si l'historique approche de sa part du budget (M2)
   │      les plus vieux tours → résumé structuré, sauvegardé en base
   │
   ├─► ContextBuilder.build(profil, historique non résumé, résumé)
   │      1. prompt système + profil          (toujours inclus)
   │      2. résumé des anciens tours         (part plafonnée du budget)
   │      3. documents RAG                    (part plafonnée du budget)
   │      4. historique, du plus récent au plus ancien, tant qu'il reste de la place
   │
   ├─► Ollama /api/chat en streaming → affichage progressif
   │
   └─► sauvegarde de la réponse + nombre réel de tokens (eval_count)
```

---

## 3. Le budget de tokens (`context.py`)

### Calcul du budget

```
budget = num_ctx × (1 − response_reserve)
```

Avec la config par défaut : `8192 × (1 − 0.2) = 6553` tokens pour le prompt. Les 20 % restants sont **réservés à la réponse** du modèle.

### Ordre de priorité

| Priorité | Contenu | Règle |
|---|---|---|
| 1 | Prompt système + profil apprenant | Toujours inclus |
| 2 | Résumé des anciens tours (M2) | Au plus 10 % du budget (`summary_share`) |
| 3 | Documents RAG | Au plus 25 % du budget (`rag_share`) |
| 4 | Historique non résumé | Ce qui reste, en partant du message le plus récent |

La **part de l'historique** (`history_budget`) est ce qui reste une fois que toutes les autres sections ont reçu leur part maximale. C'est cette valeur que le compresseur surveille.

Règles de l'historique :

- on parcourt les messages **du plus récent au plus ancien** et on s'arrête dès qu'un message ne tient plus ;
- le **message actuel de l'utilisateur est toujours gardé**, même s'il dépasse à lui seul ;
- les messages écartés sont signalés dans le terminal (`[N anciens messages hors contexte]`).

En temps normal, la compression (section 6) empêche l'historique de déborder. Écarter les vieux messages n'est plus que le **plan de secours**, quand la compression échoue.

### Rapport de contexte

Chaque construction produit un `ContextReport`, visible avec la commande `/context` :

```
Budget 594/960 tokens (estimés) — système+profil 103, résumé 74/96, RAG 0, historique 417/761 (13 messages gardés, 0 écartés)
Compressions : 2/2 réussies, 16 messages résumés, ~445 tokens économisés, 1 ms en moyenne
```

Il sert à déboguer et à produire des métriques pour la keynote.

### Où sont placés les documents RAG

Les morceaux renvoyés par le RAG sont ajoutés **à la fin du message système**, sous un titre `## Documents de référence`. Ils ne sont pas stockés dans l'historique : ils sont recalculés à chaque tour en fonction du dernier message.

---

## 4. Estimation des tokens (`memory/tokens.py`)

Ollama n'a **pas de point d'accès pour compter les tokens** avant l'envoi. On estime donc :

| Type de texte | Règle |
|---|---|
| Écriture latine (français, anglais, espagnol…) | 1 token ≈ 3,5 caractères (volontairement prudent) |
| Japonais, chinois, coréen | 1 token par caractère |
| Chaque message | + 4 tokens pour le formatage du chat |

**Pourquoi un cas spécial pour le japonais, le chinois et le coréen ?** La règle habituelle « 4 caractères par token » sous-estime fortement ces langues. Pour un tuteur de langues, ce serait un dépassement de budget assuré.

### Vrais nombres de tokens

Après chaque réponse, Ollama renvoie dans le dernier morceau du stream :

- `prompt_eval_count` : tokens du prompt traités ;
- `eval_count` : tokens générés.

Le nombre réel `eval_count` est enregistré avec chaque réponse (colonne `messages.tokens`), et `ContextBuilder` l'utilise à la place de l'estimation quand il existe.

> ⚠️ `prompt_eval_count` n'est **pas fiable** pour mesurer la taille totale du prompt : quand Ollama réutilise son cache, il ne compte que les tokens nouvellement traités. Ne pas s'en servir pour recalibrer l'estimation sans filtrer ces cas.

---

## 5. Mémoire persistante — M1 (`memory/store.py`)

Tout est stocké dans une base SQLite (`data/sensai.db` par défaut, dossier exclu de git).

### Schéma

```sql
sessions (id, user, title, created_at, updated_at)
messages (id, session_id → sessions, role, content, tokens, created_at)
profile  (user, key, value, updated_at)   -- clé primaire (user, key)
```

- **sessions** : une conversation. Son titre est le début du premier message de l'utilisateur.
- **messages** : chaque message (`user` / `assistant`), avec son nombre de tokens.
- **profile** : des paires clé/valeur libres par apprenant (`langue_cible`, `niveau`, `langue_maternelle`, `objectif`…).

### Pourquoi SQLite plutôt qu'un fichier JSON

- C'est dans la bibliothèque standard de Python (aucune dépendance).
- Les écritures sont atomiques : un crash ne corrompt pas l'historique.
- La **même base** accueillera la mémoire structurée de M3 (vocabulaire, erreurs) : tout ce qu'on sait d'un apprenant est au même endroit.

### Injection du profil

À chaque tour, le profil est ajouté au message système :

```
Tu es Sensai, un tuteur de langues bienveillant…

## Profil de l'apprenant
- langue_cible: japonais
- niveau: débutant A1
```

L'apprenant n'a donc jamais à répéter sa langue ou son niveau, même dans une nouvelle session.

### Plusieurs apprenants

Le paramètre `--user` (ou `"user"` dans `config.json`) sépare les profils et les sessions de chaque apprenant.

---

## 6. Compression de l'historique — M2 (`memory/compressor.py`)

Quand une leçon dure, l'historique finit par dépasser sa part du budget. Au lieu d'écarter les plus vieux messages (et de perdre ce que l'apprenant a appris ou raté), on les **remplace par un résumé structuré**.

### Quand la compression se déclenche

Avant chaque envoi, on compare la taille de l'historique non résumé à sa part du budget (`history_budget`) :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `compress_trigger` | `0.8` | On compresse quand l'historique dépasse 80 % de sa part |
| `compress_keep` | `0.4` | Les tours les plus récents, jusqu'à 40 % de la part, restent tels quels |
| `summary_share` | `0.1` | Taille maximale du résumé : 10 % du budget total |

Le dernier message (celui de l'utilisateur) n'est jamais compressé. Il faut au moins 2 messages à résumer, sinon on ne fait rien.

Une passe résume au plus 50 % de la part de l'historique. Une longue session rechargée avec `/load` est donc compressée **en plusieurs passes**, pour que la requête de résumé tienne elle-même dans la fenêtre du modèle.

La commande `/compress` force une compression immédiate (pratique pour la démo).

### Le format du résumé

Le modèle produit du **JSON imposé par un schéma** (paramètre `format` d'Ollama, `temperature` à 0). Les champs sont pensés pour l'apprentissage d'une langue :

| Champ | Type | Contenu | Fusion |
|---|---|---|---|
| `apprenant` | liste | Préférences, objectifs, infos personnelles | Ajout à la suite |
| `vocabulaire` | liste | « mot = traduction » | Ajout à la suite |
| `erreurs` | liste | « erreur → correction (règle) » | Ajout à la suite |
| `exercices` | liste | Exercices faits et résultat | Ajout à la suite |
| `resume` | texte | 2 à 3 phrases sur l'extrait | Remplacé |
| `en_cours` | texte | Exercice non terminé, prochaine étape | Remplacé |

Le résumé est injecté dans le message système, en Markdown :

```
## Résumé des échanges précédents
Camille a travaillé les particules…
### Apprenant
- veut voyager à Kyoto en avril
### Vocabulaire vu
- inu = chien
### Erreurs à surveiller
- confond wa et ga
### En cours
QCM sur les couleurs
```

### Choix techniques

**Le modèle ne résume que le nouveau morceau. C'est le code qui fusionne.**
L'approche naïve consiste à envoyer « ancien résumé + nouveaux messages » et à demander un résumé complet réécrit. Avec un modèle 4B, chaque réécriture risque de perdre des faits, et les pertes s'accumulent de passe en passe. Ici :
- le modèle reçoit seulement les messages à résumer, plus le texte `resume` précédent comme contexte ;
- le code **ajoute** les nouveaux éléments aux listes existantes, en supprimant les doublons (sans tenir compte des majuscules ni des espaces).

Un fait déjà résumé ne peut donc disparaître que par une règle explicite et prévisible (voir ci-dessous).

**Le JSON plutôt que du texte libre.**
- On peut vérifier et fusionner le résultat champ par champ.
- Les champs obligent le modèle à chercher ce qui compte pour un apprenant (mots, erreurs) plutôt qu'à paraphraser la conversation.
- Les mêmes champs serviront de base à M3 (vocabulaire et erreurs en base de données).

**Un plafond strict, appliqué par le code.**
Si le résumé dépasse `summary_share`, on retire les éléments **les plus anciens**, dans cet ordre : exercices, puis vocabulaire, puis erreurs, puis faits sur l'apprenant. En dernier recours, le texte `resume` est raccourci. Pas de nouvel appel au modèle, donc pas de surprise. Une fois M3 en place, le vocabulaire et les erreurs seront de toute façon conservés en base.

### Stockage

```sql
summaries    (session_id, content, covered_until, tokens, updated_at)
compressions (id, session_id, messages_in, tokens_before, tokens_after, duration_ms, ok, error, created_at)
```

- `summaries` : un résumé par session (le JSON). `covered_until` est l'id du dernier message résumé. Avec `/load`, on recharge le résumé **et seulement les messages qui suivent**.
- Les messages résumés **restent en base** : on ne perd jamais l'historique brut, il n'est simplement plus envoyé au modèle.
- `compressions` : un journal de chaque passe, réussie ou non. Il alimente les statistiques de `/context` et les métriques de la keynote.

### En cas d'échec

Si Ollama ne répond pas ou renvoie du JSON invalide :
1. l'échec est enregistré dans `compressions` (avec le message d'erreur) et signalé dans le terminal ;
2. `ContextBuilder` revient au plan de secours : les plus vieux messages sont écartés ;
3. on attend 3 tours avant de réessayer, pour ne pas ralentir chaque message si le modèle échoue en boucle.

### Mesurer : le banc de test de rappel

Le catalogue précise que la note de M2 « dépend des métriques ». Le script `benchmarks/m2_recall.py` mesure si la compression conserve vraiment les faits :

1. il place 6 faits au **début** d'une leçon synthétique (ville du voyage, mot appris, particules confondues, type d'exercice préféré…) ;
2. il ajoute N échanges de remplissage pour pousser ces faits hors du contexte ;
3. il pose une question par fait, avec trois stratégies :

| Stratégie | Description |
|---|---|
| `full` | Grande fenêtre, rien n'est retiré (référence haute) |
| `truncate` | Petite fenêtre, les vieux messages sont écartés (sans M2) |
| `compress` | Petite fenêtre, les vieux messages sont résumés (avec M2) |

```bash
.venv/bin/python -m benchmarks.m2_recall --model qwen3.8:4b --num-ctx 2048 --filler 30 --json results.json
```

Il affiche un tableau Markdown (rappel, tokens de prompt moyens, nombre et durée des compressions), prêt à coller dans la keynote. Un fait est compté comme retrouvé si la réponse contient un des mots-clés attendus.

---

## 7. Utilisation

### Lancer

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama serve                                  # dans un autre terminal
.venv/bin/python -m sensai                    # config.json par défaut
.venv/bin/python -m sensai --model qwen3:4b --num-ctx 16384 --user alice
```

### Commandes

| Commande | Effet |
|---|---|
| `/help` | Liste des commandes |
| `/new` | Nouvelle session |
| `/sessions` | Liste des sessions de l'apprenant (`*` = session active) |
| `/load <id>` | Reprend une session avec tout son historique |
| `/profile` | Affiche le profil |
| `/profile set <clé> <valeur>` | Ajoute ou modifie une info (la valeur peut contenir des espaces) |
| `/profile del <clé>` | Supprime une info |
| `/context` | Répartition du budget lors du dernier envoi + statistiques de compression |
| `/summary` | Affiche le résumé des anciens échanges |
| `/compress` | Force la compression des anciens échanges |
| `/quit` | Quitter (Ctrl+D marche aussi) |

Ctrl+C pendant une réponse coupe la génération. La partie déjà reçue est conservée.

### Configuration (`config.json`)

| Clé | Défaut | Rôle |
|---|---|---|
| `model` | `qwen3.8:4b` | Tag Ollama du modèle |
| `host` | `http://localhost:11434` | Adresse du serveur Ollama |
| `num_ctx` | `8192` | Taille de la fenêtre de contexte (tokens) |
| `response_reserve` | `0.2` | Part de `num_ctx` réservée à la réponse |
| `think` | `false` | Mode « thinking » de Qwen3 (`false` = plus rapide, `null` = ne pas envoyer le paramètre si le modèle ne le gère pas) |
| `db_path` | `data/sensai.db` | Base SQLite |
| `user` | `default` | Apprenant actif |
| `system_prompt` | tuteur façon Duolingo | Rôle du modèle |
| `compression` | `true` | Active la compression automatique (M2) |
| `compress_trigger` | `0.8` | Seuil de déclenchement (part de l'historique) |
| `compress_keep` | `0.4` | Part de l'historique gardée telle quelle (doit être < `compress_trigger`) |
| `summary_share` | `0.1` | Taille maximale du résumé (part du budget) |

Les arguments `--model`, `--host`, `--num-ctx`, `--db`, `--user` et `--config` remplacent les valeurs du fichier.

### Gestion des erreurs

| Situation | Comportement |
|---|---|
| Ollama arrêté au démarrage | Message clair, sortie avec le code 1 |
| Modèle absent | Message avec la commande `ollama pull` à lancer |
| Connexion perdue pendant une réponse | Message d'erreur, le chat continue |
| Compression échouée (JSON invalide, Ollama indisponible) | Message, vieux messages écartés, nouvel essai 3 tours plus tard |
| Entrée vide | Ignorée |
| `config.json` invalide ou clé inconnue | Message d'erreur, sortie avec le code 2 |

---

## 8. Interface avec le RAG

Le RAG (géré par un coéquipier) se branche sur `ContextBuilder` via une seule fonction :

```python
def retrieve(query: str, max_tokens: int) -> list[str]:
    """Renvoie des morceaux de documents dont le total tient dans max_tokens."""

ContextBuilder(system_prompt, num_ctx, response_reserve, retriever=retrieve, rag_share=0.25)
```

Le contrat :

- `query` est le dernier message de l'utilisateur ;
- `max_tokens` = `budget × rag_share` : **c'est au RAG de respecter cette limite** (il peut utiliser `estimate_tokens` de `memory/tokens.py` pour compter de la même façon) ;
- renvoyer une liste vide si rien n'est pertinent : aucune place n'est alors consommée.

---

## 9. Prochaines étapes

### M3 — Mémoire structurée de l'apprenant

Nouvelles tables dans la même base :

```sql
vocabulary (user, word, translation, language, correct, wrong, next_review, ...)
mistakes   (user, pattern, example, count, last_seen, ...)
```

- **Écriture :** après chaque échange (ou à chaque compression, en réutilisant les champs `vocabulaire` et `erreurs` du résumé), le modèle renvoie en JSON (paramètre `format` d'Ollama avec un schéma) des opérations : ajouter un mot, noter une bonne ou mauvaise réponse, enregistrer une erreur. Le code valide puis applique ces opérations. C'est l'agent qui pilote ses propres créations, lectures, mises à jour et suppressions, comme le demande le sujet.
- **Révisions espacées :** `next_review` est recalculé selon les bonnes et mauvaises réponses, comme chez Duolingo.
- **Lecture :** `ContextBuilder` injecte les mots à réviser aujourd'hui et les erreurs les plus fréquentes (part dédiée d'environ 10 % du budget). Le tuteur peut ainsi construire des exercices ciblés.

Répartition du budget visée une fois M3 en place :

| Contenu | Part |
|---|---|
| Prompt système + profil | fixe |
| Mémoire structurée (M3) | ~10 % |
| Documents RAG | ~25 % |
| Résumé des anciens tours (M2) | ~10 % |
| Tours récents | le reste |
| Réserve pour la réponse | 20 % de `num_ctx` |

Ces parts sont un point de départ, à ajuster avec de vraies mesures.

---

## 10. Limites actuelles

- Rien n'a encore été testé avec le vrai modèle, seulement avec un faux serveur Ollama. Les chiffres de rappel restent à mesurer avec `benchmarks/m2_recall.py`.
- La qualité du résumé dépend du modèle : un 4B peut rater un fait ou en mal formuler un. Le schéma JSON et la fusion par le code limitent les dégâts sans les supprimer.
- La compression ajoute un appel au modèle, donc quelques secondes d'attente sur le tour où elle se déclenche.
- La suppression des doublons est textuelle : « neko = chat » et « neko = le chat » restent deux éléments.
- L'estimation des tokens est approximative. La réserve de 20 % sert de marge de sécurité.
- Le profil est rempli à la main via `/profile`. Avec M3, le modèle pourra le compléter lui-même.
