# Mémoire & Contexte — fonctionnement

Ce document explique comment Sensai gère la **mémoire** (ce qu'il retient entre les messages et entre les sessions) et le **contexte** (ce qu'on envoie réellement au modèle à chaque tour).

Cas d'usage fil rouge : un **tuteur de langues façon Duolingo**, sur un modèle local Ollama (par défaut `qwen3.8:4b`, tag à vérifier avec `ollama list`).

Features du catalogue concernées :

| ID | Feature | État |
|---|---|---|
| M1 | Session Persistence & Profiles | ✅ fait |
| M2 | Token Budgeting & Semantic Compression | 🟡 budget fait, compression à faire |
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
    store.py          # SQLite : sessions, messages, profil (M1) — accueillera M3
    tokens.py         # estimation du nombre de tokens
```

Déroulement d'un tour :

```
saisie utilisateur
   │
   ├─► sauvegarde du message en base (store.add_message)
   │
   ├─► ContextBuilder.build(profil, historique)
   │      1. prompt système + profil          (toujours inclus)
   │      2. documents RAG                    (part plafonnée du budget)
   │      3. historique, du plus récent au plus ancien, tant qu'il reste de la place
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
| 2 | Documents RAG | Au plus 25 % du budget (`rag_share`) |
| 3 | Historique | Ce qui reste, en partant du message le plus récent |

Règles de l'historique :

- on parcourt les messages **du plus récent au plus ancien** et on s'arrête dès qu'un message ne tient plus ;
- le **message actuel de l'utilisateur est toujours gardé**, même s'il dépasse à lui seul ;
- les messages écartés sont signalés dans le terminal (`[N anciens messages hors contexte]`).

Pour l'instant, les messages qui ne tiennent plus sont **écartés**. M2 les remplacera par un **résumé** (voir section 7).

### Rapport de contexte

Chaque construction produit un `ContextReport`, visible avec la commande `/context` :

```
Budget 131/6553 tokens (estimés) — système+profil 124, RAG 0, historique 7 (1 messages gardés, 0 écartés)
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

## 6. Utilisation

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
| `/context` | Répartition du budget lors du dernier envoi |
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

Les arguments `--model`, `--host`, `--num-ctx`, `--db`, `--user` et `--config` remplacent les valeurs du fichier.

### Gestion des erreurs

| Situation | Comportement |
|---|---|
| Ollama arrêté au démarrage | Message clair, sortie avec le code 1 |
| Modèle absent | Message avec la commande `ollama pull` à lancer |
| Connexion perdue pendant une réponse | Message d'erreur, le chat continue |
| Entrée vide | Ignorée |
| `config.json` invalide ou clé inconnue | Message d'erreur, sortie avec le code 2 |

---

## 7. Interface avec le RAG

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

## 8. Prochaines étapes

### M2 — Compression de l'historique

Au lieu d'écarter les vieux messages :

1. quand l'historique dépasse sa part du budget, prendre les plus vieux tours ;
2. demander au modèle un **résumé court** qui garde les faits utiles à l'apprentissage (erreurs commises, mots vus, exercices faits, préférences exprimées) ;
3. stocker ce résumé en base et l'injecter dans le contexte à la place des tours résumés (part dédiée d'environ 10 % du budget) ;
4. re-résumer le résumé s'il grossit trop.

**Métriques à produire** (le catalogue dit « dépend des métriques ») : un test de rappel. On place des faits dans une longue conversation, on force la compression, puis on demande au modèle de les restituer. On compare le taux de rappel avec et sans compression, et on mesure les tokens économisés.

Point d'attention : un modèle 4B résume de façon approximative. Il faudra un prompt de résumé très cadré, idéalement avec une sortie JSON imposée.

### M3 — Mémoire structurée de l'apprenant

Nouvelles tables dans la même base :

```sql
vocabulary (user, word, translation, language, correct, wrong, next_review, ...)
mistakes   (user, pattern, example, count, last_seen, ...)
```

- **Écriture :** après chaque échange, le modèle renvoie en JSON (paramètre `format` d'Ollama avec un schéma) des opérations : ajouter un mot, noter une bonne ou mauvaise réponse, enregistrer une erreur. Le code valide puis applique ces opérations. C'est l'agent qui pilote ses propres créations, lectures, mises à jour et suppressions, comme le demande le sujet.
- **Révisions espacées :** `next_review` est recalculé selon les bonnes et mauvaises réponses, comme chez Duolingo.
- **Lecture :** `ContextBuilder` injecte les mots à réviser aujourd'hui et les erreurs les plus fréquentes (part dédiée d'environ 10 % du budget). Le tuteur peut ainsi construire des exercices ciblés.

Répartition du budget visée une fois M2 et M3 en place :

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

## 9. Limites actuelles

- Les vieux messages sont **écartés**, pas encore résumés (en attente de M2).
- L'estimation des tokens est approximative. La réserve de 20 % sert de marge de sécurité.
- Rien n'a encore été testé avec le vrai modèle, seulement avec un faux serveur Ollama.
- Le profil est rempli à la main via `/profile`. Avec M3, le modèle pourra le compléter lui-même.
