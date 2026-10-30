# Reproduire les mesures et lire les logs

## Vérifications hors ligne, sans installation

Depuis la racine du dépôt, avec Python 3.10 ou plus :

```bash
python -B tools/verify_evidence.py
python -B tools/recalculate_benchmarks.py
```

Ces commandes vérifient les empreintes et recalculent les **248 couples image/modèle archivés**, sans GPU, réseau ni appel facturé. Les deux études imprimées réutilisent les 31 sorties Unlimited ; 248 lignes ne représentent donc pas 248 inférences indépendantes. Les contrôles comparent également les agrégats publiés.

Le 30 octobre, **60 tests portables** sur les annotations, le flux, les tableaux et la fidélité ont passé, ainsi que le recalcul complet des scores. Voir [le compte rendu](../../results/verification-20261030.json). La suite complète n’est pas validée sur le poste Windows : pytest, boto3/botocore, markdown-it et le module Unix `fcntl` n’y sont pas disponibles. Aucun paquet n’a été installé pour produire ce dossier.

Pour la suite historique complète, utiliser l’environnement Linux documenté, ajouter `scripts/` au chemin Python, vérifier d’abord la présence de pytest, puis :

```bash
python -m pytest --version
PYTHONPATH=scripts python -m pytest -q tests
```

Ils couvrent les flux de données, la reprise, la conversion des tableaux et la fidélité. Ils ne remplacent pas un test GPU de l’entraînement ni une validation de qualité sur des documents inconnus.

## Les fichiers à consulter

| Question | Fichier |
|---|---|
| Le calcul a-t-il vraiment fini ? | `results/training/markdown-v3/verified-final.json` |
| Tous les pas sont-ils là ? | `results/training/markdown-v3/metrics-complete.jsonl` : pas 1 à 4796 |
| Train loss et LR ? | `results/training/markdown-v3/losses-complete.csv` |
| Validation tableaux / contrôle ? | `results/training/markdown-v3/evaluation-complete.csv` |
| Logs de départ et reprise ? | `results/logs/`, index `inventory.json` |
| D’où vient l’ancien graphe du 20 octobre ? | `results/training/markdown-v3/snapshot-20261020/`, photographie partielle datée |
| Sorties exactes des modèles ? | Chaque `results/benchmarks/*/resultats.json`, clé `cases[].outputs` |
| Références et provenance des pages ? | `jeu.json` et `cases[].reference` dans le même dossier |
| Essai correctif ? | `results/training/corrective-v1/` : config, événements, checkpoints JSON et générations |

Les logs Kubernetes ont été relus le 30 octobre, sans modifier les workloads. Les URLs et valeurs ressemblant à des accès ont été expurgées avant transfert. Les lignes numériques de métriques sont conservées. **Le journal de démarrage contient un échec historique et le journal de reprise une réussite : les deux sont nécessaires.**

Les 202 événements d’évaluation comprennent les évaluations réellement émises, y compris des évaluations proches lors d’arrêts/reprises. Le graphe ne fabrique pas des mesures entre ces points. La courbe train utilise les 4 796 mises à jour ; le lissage, s’il est affiché, est explicitement identifié.

## Réexécuter l’inférence ou l’entraînement

Le recalcul CER/WER est entièrement portable avec les textes archivés. Une nouvelle inférence exige les images, les poids et un environnement compatible ; un entraînement exige en plus l’accès au corpus massif et aux états de reprise. Les poids ne sont pas dans GitHub.

`experiments/markdown-v3/code/` conserve les sources effectivement embarquées et leur SHA. `experiments/corrective-v1/code/` conserve le trainer correctif dont les trois sources critiques correspondent à la configuration de fin. Les chemins `/work`, `/export`, montages, bibliothèques CUDA et Secrets de ces scripts appartiennent à leur exécution d’origine ; ce ne sont pas des commandes universelles à copier sur un autre cluster. `final_validation.py` est une source d’audit avec identités attendues à fournir, pas un lanceur autonome.

Les clients dans `evaluation/` sont des sources d’exécution conservées. Les endpoints, entrées et accès doivent être configurés dans un nouvel environnement ; leur simple import ne constitue pas un essai terminé. Pour des appels API, fournir la clé uniquement par variable d’environnement ou saisie masquée. Le dépôt ne relance automatiquement ni entraînement, ni service, ni suivi périodique.

## Traçabilité et licences

`results/export-provenance.json` relie les copies aux fichiers sources locaux et à leur SHA. `EVIDENCE-SHA256.json` vérifie l’instantané diffusé. Un hash garantit l’identité d’un fichier, pas la justesse de sa transcription.

Les références et sorties incluses proviennent de benchmarks publics. Voir [les notices](../../DATA-NOTICES.md) avant toute réutilisation : plusieurs corpus sont sous licence non commerciale. Les scans originaux et le corpus privé d’entraînement ne sont pas republiés ; les rapports contiennent des illustrations de pages avec leurs attributions.
