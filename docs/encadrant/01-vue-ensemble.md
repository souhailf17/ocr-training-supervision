# Vue d’ensemble pour l’encadrant

État documenté au **30 octobre 2026**. Les chiffres viennent de fichiers et de logs conservés ; ce document ne lance aucun calcul.

## Objectif

Adapter Qwen3.5-9B à la transcription de documents en arabe, français et anglais, avec mise en forme Markdown et tableaux, puis vérifier la fidélité du texte produit. L’objectif métier est une extraction exploitable dans Sovereign Document Suite : lire ce qui est visible, conserver signes et chiffres, ne pas ajouter de texte après la page.

## Ce qui a été fait

```mermaid
flowchart LR
  A[Images et annotations sources] --> B[Contrôle intégrité et splits]
  B --> C[Reconstruction des tableaux Markdown]
  C --> D[Qwen de base gelé + LoRA]
  D --> E[Loss de validation et checkpoints]
  E --> F[Génération sur images sans référence]
  F --> G[CER, WER, lecture des erreurs]
  G --> H[Choix et corrections à valider]
```

1. Préparation et entraînement historiques, jusqu’au checkpoint parent `14355`.
2. Correction des cibles : les renvois internes de tableaux sont remplacés par leur contenu réel. Une nouvelle passe sur originaux + `aug1` conserve les splits et commence avec un optimiseur neuf.
3. Passe Markdown sur un H200, terminée au pas de phase `4796` le 26 octobre. Reprise contrôlée après un arrêt lié au stockage au pas `4682`.
4. Comparatif exploratoire de 31 images imprimées : notre checkpoint **4707**, Qwen de base, dots.mocr, puis Unlimited OCR 2. Le checkpoint final **4796 n’est pas la colonne évaluée dans ce comparatif**.
5. Diagnostic de fidélité du modèle final : répétitions, signes remplacés et confusion d’alphabets de chiffres. Un petit entraînement correctif indépendant est exécuté depuis le Qwen original avec une LoRA neuve.
6. Comparatifs Mistral / Unlimited : 31 images imprimées, puis 31 images manuscrites réelles. Les deux corpus sont distincts.

## Situation actuelle

| Élément | État prouvé | Interprétation |
|---|---|---|
| Passe Markdown | 1 841 540 / 1 841 540 exemples, checkpoint 4796 vérifié PVC/MinIO | Le calcul a fini ; cela ne valide pas la qualité OCR |
| Comparatif Qwen | 31 sorties du checkpoint 4707 et 31 du modèle de base | La fine-tune est moins bonne en moyenne globale sur ce jeu |
| Essai correctif | 38 mises à jour, 600 exemples consommés, 300 pages uniques | Loss plus basse mais erreurs de chiffres/signes encore présentes ; candidat non promu |
| Imprimés Mistral / Unlimited | 62 résultats réellement obtenus | Mistral a le plus faible CER/WER global sur ce corpus |
| Manuscrits Mistral / Unlimited | 62 résultats réellement obtenus | Mistral meilleur en moyenne ; les deux restent faibles sur l’arabe historique |

## Ce qu’il faut regarder en réunion

Commencer par les [résultats comparés](03-resultats-et-limites.md), puis consulter les sorties complètes par page dans les JSON des benchmarks. Lire le [bilan correctif](../../reports/corrective/BILAN.md) et le graphe de fidélité : une loss plus faible n’a pas éliminé les erreurs de copie. Les [questions proposées](05-decisions-a-prendre.md) visent à choisir la prochaine expérience, pas à justifier automatiquement un entraînement plus long.

Le dépôt contient le code, les logs expurgés, les configurations, les sorties de benchmarks publics et les références nécessaires au recalcul. Les poids, le corpus massif d’entraînement et les accès de production restent hors GitHub.
