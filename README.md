# OCR multilingue — entraînement, diagnostics et comparaisons

**Dossier de travail pour l’encadrant · état vérifié au 30 octobre 2026**

Ce dépôt rassemble le code d’adaptation de **Qwen3.5-9B**, les configurations utilisées, les logs d’entraînement, les mesures et les comparatifs avec **Qwen de base, dots.mocr, Unlimited OCR 2 et Mistral OCR 4.1**. L’objectif est de comprendre ce qui fonctionne, les régressions et les expériences à décider ensemble.

## Commencer ici

| Lecture | Contenu |
|---|---|
| **[Vue d’ensemble](docs/encadrant/01-vue-ensemble.md)** | Objectif, étapes réalisées et état de chaque modèle |
| [Données et entraînement](docs/encadrant/02-donnees-et-entrainement.md) | Corpus, paramètres, pertes, reprise et checkpoints |
| **[Résultats et limites](docs/encadrant/03-resultats-et-limites.md)** | Comparaisons, hallucinations, erreurs et interprétation |
| [Reproduction et logs](docs/encadrant/04-reproduire-et-lire-les-logs.md) | Recalcul hors ligne et chemin vers chaque preuve |
| **[Décisions à discuter](docs/encadrant/05-decisions-a-prendre.md)** | Questions concrètes et prochaine expérimentation |

## État de l’entraînement

La passe Markdown principale est **terminée et vérifiée**, après une reprise contrôlée le 26 octobre : **1 841 540 / 1 841 540 exemples**, **4 796 mises à jour**, un H200, contexte 8 192, LoRA 16/32. La dernière train loss est **0,135428**, la validation **0,159622** sur le monitoring de 256 pages.

Le checkpoint final est **4796** (compteur global 19151). [Preuve indépendante et SHA-256](results/training/markdown-v3/verified-final.json). Cela atteste la fin du calcul et l’intégrité du fichier ; cela ne garantit pas une transcription fidèle.

![Toutes les mises à jour et évaluations de la passe Markdown](reports/training/markdown-losses-complete.png)

Les [logs expurgés](results/logs/) et les [4 796 mesures complètes](results/training/markdown-v3/losses-complete.csv) sont disponibles. L’ancien instantané partiel du 20 octobre est conservé et identifié comme tel.

## Résultats principaux

Les taux sont des pourcentages d’erreur : **plus faible = meilleur**. Il s’agit de micro moyennes ; toutes les sorties, y compris les dérives, sont conservées.

| Expérience | Modèle | CER % | WER % |
|---|---|---:|---:|
| 31 imprimés, comparatif historique | Notre Qwen **4707** | 95,18 | 109,96 |
| Même jeu imprimé | Qwen original | 61,59 | 72,87 |
| Même jeu imprimé | dots.mocr | 571,22 | 19,25 |
| Même jeu imprimé | Unlimited OCR 2 | 7,48 | 14,75 |
| Même jeu imprimé, étude suivante | Mistral OCR 4.1 | **5,90** | **11,56** |
| 31 manuscrits, corpus distinct | Mistral OCR 4.1 | **56,57** | **75,18** |
| Même jeu manuscrit | Unlimited OCR 2 | 250,39 | 340,50 |

**Attention aux identités :** le comparatif Qwen utilise le checkpoint **4707**, encore intermédiaire ; il ne mesure pas le final **4796**. Le candidat correctif est une troisième adaptation, distincte, et n’a pas remplacé cette colonne. Les mêmes sorties Unlimited imprimées sont réutilisées entre les études imprimées.

Un CER supérieur à 100 % est possible à cause d’insertions et de répétitions. La dérive de ponctuation de dots.mocr sur une page est conservée. Les deux modèles du comparatif manuscrit restent en difficulté sur l’arabe historique. Ni un faible CER moyen, ni une fin EOS, ni une baisse de loss ne prouvent l’absence d’hallucinations.

- **[Rapport imprimés Mistral / Unlimited — PDF](reports/printed/rapport-scientifique.pdf)** · [Markdown](reports/printed/rapport-scientifique.md)
- **[Rapport manuscrits — PDF](reports/handwriting/rapport-manuscrit.pdf)** · [Markdown](reports/handwriting/rapport-manuscrit.md)
- [124 sorties du comparatif de quatre modèles](results/benchmarks/printed-four-models/resultats.json)
- [62 sorties imprimées Mistral / Unlimited](results/benchmarks/printed-mistral-unlimited/resultats.json)
- [62 sorties manuscrites Mistral / Unlimited](results/benchmarks/handwritten-mistral-unlimited/resultats.json)

Chaque JSON comprend les textes de référence, les sorties intégrales, les scores par image, les raisons d’arrêt disponibles et les métadonnées du protocole. Les CSV voisins permettent de trier et analyser les résultats.

## Problèmes observés et essai correctif

Le diagnostic retrouve des répétitions, du texte ajouté, des confusions de signes spéciaux et d’alphabets de chiffres. Un essai correctif depuis le Qwen de base avec une LoRA neuve a terminé ses **38 mises à jour**, sur 300 pages d’entraînement parcourues deux fois et 75 de validation. La validation baisse de **0,641595 à 0,563570** (−12,16 %), mais des erreurs persistent. **Le candidat n’est pas validé pour promotion.**

Lire le **[bilan correctif](reports/corrective/BILAN.md)**, les [générations conservées](results/training/corrective-v1/generations/) et les [questions pour l’encadrant](docs/encadrant/05-decisions-a-prendre.md).

## Organisation du dépôt

```text
docs/encadrant/        parcours de lecture et décisions à prendre
experiments/          sources effectivement embarquées et essai correctif
scripts/              préparation, variantes de trainers et outils historiques
tests/                contrôles du pipeline
results/training/     configurations, événements, pertes et preuves de fin
results/logs/         journaux expurgés et inventaire
results/benchmarks/   références, sorties et CER/WER par image
evaluation/           code des métriques et sources des clients d’inférence
reports/              rapports scientifiques et figures
tools/                vérification et recalcul hors ligne
```

## Vérifier sans relancer de calcul GPU

Python 3.10+ suffit aux deux vérifications ci-dessous ; aucune installation, aucun accès API et aucun téléchargement :

```bash
python -B tools/verify_evidence.py
python -B tools/recalculate_benchmarks.py
```

Les scripts d’entraînement sont les sources des environnements d’origine : chemins, poids, montages et versions CUDA doivent être préparés avant toute nouvelle exécution. Le dépôt n’alloue et ne lance rien automatiquement. Voir [la reproductibilité et ses limites](docs/encadrant/04-reproduire-et-lire-les-logs.md).

## Accès, données et attributions

Aucun mot de passe, clé API, secret S3, token OpenShift ou poids de modèle n’est inclus. Le corpus massif d’entraînement et les scans originaux restent hors GitHub. Les références, sorties et illustrations des benchmarks publics restent soumises à leurs licences, dont des restrictions **non commerciales** : voir [DATA-NOTICES.md](DATA-NOTICES.md) et [SECURITY.md](SECURITY.md).

Les documents racine de `docs/` et `metrics/` conservent aussi des étapes historiques. La synthèse actuelle est celle de `docs/encadrant/`. L’ancienne page d’accueil du 2 octobre reste [archivée](docs/historique/README-20261002.md).
