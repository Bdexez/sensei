# Mémoire & Contexte — fonctionnement

Ce document explique comment Sensai gère la **mémoire** (ce qu'il retient entre les messages et entre les sessions) et le **contexte** (ce qu'on envoie réellement au modèle à chaque tour).

Cas d'usage fil rouge : un **tuteur de langues façon Duolingo**, sur un modèle local Ollama (par défaut `qwen3.8:4b`, tag à vérifier avec `ollama list`).

Features du catalogue concernées :

| ID | Feature | État |
|---|---|---|
| M1 | Session Persistence & Profiles | ✅ fait |
| M2 | Token Budgeting & Semantic Compression | ✅ fait (métriques à mesurer avec le vrai modèle) |
| M3 | External Structured Memory | ✅ fait (métriques à mesurer avec le vrai modèle) |
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
    store.py          # SQLite : sessions, messages, profil (M1), résumés (M2)
    tokens.py         # estimation du nombre de tokens
    compressor.py     # compression des vieux tours en résumé structuré (M2)
    learner.py        # vocabulaire, erreurs, journal des opérations (M3)
    extractor.py      # le modèle choisit les opérations mémoire après chaque échange (M3)
benchmarks/
  m2_recall.py        # banc de test : rappel des faits avec / sans compression
  m3_extraction.py    # banc de test : qualité des opérations mémoire proposées
```

Déroulement d'un tour :

```
saisie utilisateur
   │
   ├─► application des opérations mémoire de l'échange précédent (M3)
   │
   ├─► sauvegarde du message en base (store.add_message)
   │
   ├─► compression si l'historique approche de sa part du budget (M2)
   │      les plus vieux tours → résumé structuré, sauvegardé en base
   │
   ├─► ContextBuilder.build(profil, historique non résumé, résumé, mémoire)
   │      1. prompt système + profil          (toujours inclus)
   │      2. mémoire de l'apprenant (M3)      (part plafonnée du budget)
   │      3. résumé des anciens tours         (part plafonnée du budget)
   │      4. documents RAG                    (part plafonnée du budget)
   │      5. historique, du plus récent au plus ancien, tant qu'il reste de la place
   │
   ├─► Ollama /api/chat en streaming → affichage progressif
   │
   ├─► sauvegarde de la réponse + nombre réel de tokens (eval_count)
   │
   └─► extraction des opérations mémoire en arrière-plan (M3)
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
| 2 | Mémoire de l'apprenant (M3) | Au plus 10 % du budget (`memory_share`) |
| 3 | Résumé des anciens tours (M2) | Au plus 10 % du budget (`summary_share`) |
| 4 | Documents RAG | Au plus 25 % du budget (`rag_share`) |
| 5 | Historique non résumé | Ce qui reste, en partant du message le plus récent |

Avec la config par défaut (8192 tokens) : réserve pour la réponse 1639, mémoire ≤ 655, résumé ≤ 655, RAG ≤ 1638, et environ 3 500 tokens pour l'historique (environ 5 100 tant que le RAG n'est pas branché).

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
Mémoire de l'apprenant 85/96
Opérations mémoire : 6/7 appliquées, 1 extractions échouées
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

## 7. Mémoire structurée de l'apprenant — M3 (`memory/learner.py`, `memory/extractor.py`)

Le résumé de M2 garde le fil d'**une** session. M3 garde ce que l'apprenant sait **sur la durée, toutes sessions confondues** : les mots vus, quand les réviser, et les erreurs qui reviennent. C'est la « mémoire de prof » d'un Duolingo.

### Les données

```sql
vocabulary (user, language, word, translation, box, correct, wrong, next_review, created_at, updated_at)
           -- unique (user, language, word)
mistakes   (user, language, pattern, correction, example, count, last_seen)
           -- unique (user, language, pattern)
memory_ops (user, session_id, op, payload, applied, error, created_at)
```

- La langue vient du profil (`langue_cible`). Un apprenant qui étudie deux langues a deux vocabulaires séparés.
- Mots et erreurs sont stockés en minuscules, avec les espaces normalisés : « Inu » et « inu » sont le même mot, et une erreur déjà vue voit son compteur augmenter au lieu d'être dupliquée.
- `memory_ops` journalise **chaque opération proposée par le modèle**, acceptée ou rejetée, avec la raison du rejet. On peut toujours expliquer pourquoi la mémoire contient quelque chose.

### Révision espacée (système de Leitner)

Chaque mot est dans une « boîte » de 0 à 5. La boîte décide quand le mot doit être révisé :

| Boîte | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Prochaine révision | tout de suite | 1 jour | 3 jours | 7 jours | 14 jours | 30 jours |

- Bonne réponse : le mot monte d'une boîte.
- Mauvaise réponse : le mot retombe en boîte 0.
- Un mot est considéré comme **maîtrisé** à partir de la boîte 4.

Leitner plutôt que SM-2 (l'algorithme d'Anki) : il est plus simple à expliquer, il n'a besoin que de « juste / faux » (ce qu'un modèle 4B sait juger de façon fiable), et il suffit pour la démo.

### Écriture : c'est le modèle qui décide, le code qui applique

Après chaque échange, un appel séparé au modèle lit le message de l'apprenant, la réponse du tuteur, le profil, ainsi que les mots et erreurs déjà connus. Il renvoie une liste d'opérations, en JSON imposé par un schéma :

| Opération | Champs | Effet |
|---|---|---|
| `add_word` | word, translation | Ajoute un mot (ou met à jour sa traduction) |
| `review_word` | word, correct, (translation) | Fait monter ou redescendre le mot dans les boîtes de Leitner |
| `delete_word` | word | Oublie un mot |
| `add_mistake` | pattern, correction, (example) | Ajoute une erreur, ou augmente son compteur |
| `resolve_mistake` | pattern | Retire une erreur que l'apprenant maîtrise désormais |
| `set_profile` | key, value | Complète le profil (langue, niveau, objectif…) |

C'est ce qui fait de M3 une mémoire **pilotée par l'agent** : le modèle choisit lui-même de créer, modifier ou supprimer des entrées. Avant toute écriture, le code vérifie chaque opération :

- type d'opération connu, champs obligatoires présents, textes de moins de 200 caractères, `correct` bien booléen ;
- `review_word` sur un mot inconnu : accepté seulement si une traduction est fournie (le mot est alors créé) ;
- `delete_word` ou `resolve_mistake` sur quelque chose qui n'existe pas : rejeté ;
- au plus 12 opérations par échange.

Une opération invalide est **rejetée et journalisée**, sans bloquer les autres.

Pour que le modèle réutilise exactement le même texte (et que les compteurs augmentent au lieu de créer des doublons), on lui donne les mots connus présents dans l'échange et les 10 erreurs les plus fréquentes.

### Choix techniques

**Un appel séparé plutôt que des appels d'outils pendant la réponse.**
On aurait pu laisser le tuteur appeler des outils (« enregistre ce mot ») en pleine réponse. On ne l'a pas fait :
- le tuteur reste concentré sur l'enseignement, et sa réponse continue de s'afficher progressivement ;
- un modèle 4B est bien plus fiable sur une seule tâche d'extraction étroite que pour décider en pleine réponse quand appeler un outil ;
- le schéma JSON garantit une sortie lisible par le code.

**L'extraction tourne en arrière-plan.**
Elle démarre dès que la réponse est affichée, dans un fil séparé, pendant que l'apprenant lit et tape. Ses opérations sont appliquées au début du message suivant (ou en quittant). Seul l'appel au modèle tourne en arrière-plan : la base SQLite n'est touchée que par le fil principal, ce qui évite les problèmes d'accès concurrent.

**La lecture est faite par le code, pas par le modèle.**
Choisir quoi injecter est déterministe : pas besoin d'un appel au modèle pour « chercher » dans la mémoire.

### Lecture : ce qui est injecté dans le contexte

À chaque message, `LearnerMemory.render()` construit un bloc ajouté au message système. Il est rempli par priorité, ligne par ligne, jusqu'à `memory_share` du budget :

1. **Mots présents dans le message** de l'apprenant (recherche par mot entier, pour que « inu » ne soit pas trouvé dans « inutile » ; par sous-chaîne pour le japonais et le chinois, qui s'écrivent sans espaces) ;
2. **Mots à réviser** (`next_review` dépassé), les plus fragiles d'abord ;
3. **Erreurs les plus fréquentes**.

```
## Mémoire de l'apprenant (japonais)
3 mots vus, 0 maîtrisés.

### Mots présents dans le message
- inu = chien (2 ✔ / 0 ✘)

### Mots à réviser
- neko = chat (0 ✔ / 1 ✘)

### Erreurs fréquentes
- confond wa et ga → wa = thème, ga = sujet (vue 2 fois) — ex. « watashi ga Camille desu »

Réutilise les mots à réviser dans tes exercices et surveille les erreurs fréquentes.
```

La dernière ligne demande au tuteur d'utiliser cette mémoire : c'est ce qui transforme le chatbot en prof qui fait réviser au bon moment.

### Lien avec M2

Le résumé de M2 contient aussi des listes `vocabulaire` et `erreurs`. Il y a un recoupement assumé : le résumé sert au fil de **la session en cours**, M3 à la mémoire **à long terme**. Comme M3 garde tout en base, le résumé peut retirer ses vieux mots et vieilles erreurs quand il manque de place sans que rien ne soit perdu.

### En cas d'échec

Si l'extraction échoue (JSON invalide, Ollama indisponible), l'échec est journalisé (`op = 'extract'`) et signalé dans le terminal, et la conversation continue normalement. Seules les opérations de cet échange sont perdues.

### Mesurer : le banc de test d'extraction

`benchmarks/m3_extraction.py` fait tourner l'extraction sur 7 échanges écrits à la main dont on connaît les bonnes opérations (nouveau mot, bonne et mauvaise réponse, erreur de particule, infos de profil, demande d'oubli, échange sans rien à retenir). Il mesure :

- la **précision** : part des opérations proposées qui étaient attendues ;
- le **rappel** : part des opérations attendues qui ont été proposées ;
- le nombre d'opérations que la validation **rejetterait** ;
- le nombre d'**extractions échouées** (JSON invalide).

```bash
.venv/bin/python -m benchmarks.m3_extraction --model qwen3.8:4b --json m3.json
```

---

## 8. Utilisation

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
| `/vocab` | Vocabulaire appris, avec la boîte de Leitner et la date de la prochaine révision |
| `/vocab del <mot>` | Oublie un mot |
| `/mistakes` | Erreurs récurrentes, les plus fréquentes d'abord |
| `/memory` | Bloc mémoire injecté au dernier message |
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
| `memory` | `true` | Active la mémoire structurée (M3) : extraction et injection |
| `memory_share` | `0.1` | Taille maximale du bloc mémoire (part du budget). `summary_share + memory_share` doit rester ≤ 0,5 |

Les arguments `--model`, `--host`, `--num-ctx`, `--db`, `--user` et `--config` remplacent les valeurs du fichier.

### Gestion des erreurs

| Situation | Comportement |
|---|---|
| Ollama arrêté au démarrage | Message clair, sortie avec le code 1 |
| Modèle absent | Message avec la commande `ollama pull` à lancer |
| Connexion perdue pendant une réponse | Message d'erreur, le chat continue |
| Compression échouée (JSON invalide, Ollama indisponible) | Message, vieux messages écartés, nouvel essai 3 tours plus tard |
| Extraction mémoire échouée | Message, échec journalisé, la conversation continue |
| Opération mémoire invalide | Rejetée et journalisée, les autres opérations sont appliquées |
| Entrée vide | Ignorée |
| `config.json` invalide ou clé inconnue | Message d'erreur, sortie avec le code 2 |

---

## 9. Interface avec le RAG

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

## 10. Prochaines étapes

- **Mesurer** avec le vrai modèle (`m2_recall.py` et `m3_extraction.py`) et ajuster les consignes données au modèle à partir des réponses détaillées (option `--json`).
- **Régler les parts du budget** une fois le RAG branché : les valeurs actuelles (10 % mémoire, 10 % résumé, 25 % RAG) sont un point de départ.
- **M4 (document d'état)**, si le temps le permet : une fiche de progression de l'apprenant, mise à jour section par section, qui pourrait s'appuyer sur les tables de M3.

---

## 11. Limites actuelles

- Rien n'a encore été testé avec le vrai modèle, seulement avec un faux serveur Ollama. Les chiffres restent à mesurer avec `benchmarks/m2_recall.py` et `benchmarks/m3_extraction.py`.
- La qualité du résumé dépend du modèle : un 4B peut rater un fait ou en mal formuler un. Le schéma JSON et la fusion par le code limitent les dégâts sans les supprimer.
- La compression ajoute un appel au modèle, donc quelques secondes d'attente sur le tour où elle se déclenche.
- L'extraction M3 ajoute un appel au modèle à chaque échange. Il tourne en arrière-plan, mais si l'apprenant répond très vite, le message suivant attend la fin de l'extraction.
- Les erreurs sont regroupées par texte exact : si le modèle formule la même erreur de deux façons différentes, elle est comptée deux fois. Lui fournir les erreurs connues limite ce risque sans le supprimer.
- Leitner ne note que « juste / faux » : il ne distingue pas une réponse hésitante d'une réponse immédiate.
- La suppression des doublons est textuelle : « neko = chat » et « neko = le chat » restent deux éléments.
- L'estimation des tokens est approximative. La réserve de 20 % sert de marge de sécurité.
- Le modèle peut modifier le profil (`set_profile`) : une mauvaise interprétation peut écraser une valeur. Chaque changement est journalisé dans `memory_ops`, et `/profile set` permet de corriger.
