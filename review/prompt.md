Tu fais la revue de nuit de l'assistant vocal de la famille : ce qu'il a entendu, compris, fait et
répondu pendant la journée du {day}. Ton but : qu'il marche mieux demain.

Les données du jour, dans {data} :
- `interactions.json` : chaque requête (voix du satellite, voix ou texte de l'application web), avec
  ce que Parakeet a transcrit (`transcript`), une seconde transcription par Whisper large-v3 du même
  audio (`whisper`, quand il y a un enregistrement), les appels d'outils et leurs résultats
  (`failed` : rien n'a été fait), la réponse, les temps de chaque étape, le site et l'appareil.
  `[Rien compris]` : aucune parole transcrite.
- `journal.log` : le journal du serveur pour la journée (prompts, appels de fonctions, erreurs).
- `previous.md` : ce que les revues précédentes ont déjà corrigé ou proposé.

Le dépôt est ton répertoire de travail, à jour de `main`. Lis `CLAUDE.md`, puis ce dont tu as
besoin (`DECISIONS.md`, `ROADMAP.md`, le code).

## Ce que tu cherches

Lis les échanges comme la famille les a vécus, conversation par conversation : ce qui a été
demandé, ce qui a été fait, si la réponse était juste et utile, si la personne a dû répéter,
reformuler ou abandonner. Puis dans le journal, pourquoi. Pistes :
- transcription fausse (compare Parakeet et Whisper ; noms de pièces, d'enceintes, de lieux) ;
- mauvais outil, mauvais arguments, entité mal résolue, outil en échec ;
- réponse fausse, inventée, trop longue pour être dite à voix haute, ou qui ne répond pas ;
- requête répétée ou reformulée juste après : le premier essai a raté ;
- lenteur anormale (une étape bien au-dessus de ses temps habituels) ;
- erreurs et avertissements du journal.

## Ce que tu en fais

**Un bug évident, que tu corriges toi-même** : le défaut est certain (le journal le montre), la
correction est petite, locale, sans choix de comportement à faire, et un test unitaire la couvre.
Pour chacun : écris le test qui échoue, corrige, lance
`/opt/voice-assistant/venv/bin/python -m unittest test_unit` (tout doit passer), puis un commit
(`git add` des fichiers voulus, `git commit -m`), un par correction, message en anglais dans le
style de `git log` : ce qui était faux, ce que montre le journal du {day}, ce qui a été changé. Termine
le message par la ligne `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Ne pousse pas : le
script s'en charge, déploie, et reviendra en arrière si le serveur ne redémarre pas.
Trois corrections au plus par nuit.

Ne corrige jamais toi-même, propose seulement : les serrures et le déverrouillage (`lock`,
`unlock`, `confirm`), l'alarme et son désarmement (`arm_alarm`, `disarm_alarm`, `HAClient.arm`,
`HAClient.disarm`), le contrôle d'accès (`access_control`, `cross_site_guard`, clés, Funnel), les
secrets, l'enregistrement des voix, les dépendances, le firmware, et tout ce qui change le prompt
système ou le comportement de l'assistant d'une façon qui se discute.

**Une amélioration à discuter** : tout le reste qui vaut la peine. Une proposition par sujet, avec
les faits qui la motivent (heure, ce qui a été dit, ce qui s'est passé), et une ou deux pistes
concrètes. Pas de proposition sans fait du jour qui la motive, pas de proposition déjà dans
`ROADMAP.md` ou dans `previous.md`, sauf fait nouveau qui la rend plus pressante (dis-le alors).
Mieux vaut deux propositions solides que six faibles ; aucune est une bonne réponse.

Pour chaque proposition, `brief` est le message qui ouvrira une nouvelle session Claude Code dans
`~/Code/voice-assistant` pour la traiter avec Ivan : autonome (la session ne verra pas cette revue),
en français, avec les faits (citations, heures, extraits utiles du journal, fichiers et fonctions en
cause), les pistes, et ce que serait « fini ». Elle commencera par en discuter avec Ivan, pas par
coder.

## Ta réponse

En français, en Markdown sans tableau, dans les champs du JSON demandé :
- `summary` : 2 à 4 phrases sur la journée (combien de requêtes, ce qui a bien marché, ce qui a raté) ;
- `fixes` : une ligne par commit fait, ce qui était faux et ce qui est corrigé ;
- `proposals` : `title` (court), `problem` (les faits), `suggestion` (les pistes), `brief`.
