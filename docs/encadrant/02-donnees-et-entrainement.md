# Données, modèle et entraînement

## Modèle et méthode

Le modèle de base est `Qwen/Qwen3.5-9B`, révision figée `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. La base est gelée en BF16 ; seuls les adaptateurs LoRA FP32 sont entraînés : rang 16, alpha 32, **29 097 984 paramètres entraînables**. Les poids gelés ne sont pas intégralement réentraînés.

Le contexte maximal est de 8 192 tokens **image + consigne + cible**. Les images gardent leur ratio et sont limitées à 1 024 pixels sur le grand côté. Les échantillons trop longs bloquent la préparation plutôt que d’être tronqués silencieusement. Les tokens d’entrée sont masqués dans la loss ; la cible et sa terminaison sont supervisées.

## Corpus Markdown principal

Source : [OCR-Data/ocr_data](https://huggingface.co/datasets/OCR-Data/ocr_data), révision `4cf05556e21bc54aca54a8b1f81ce172f3ef0477`. Les contrôles concernent 103 archives originales et 103 archives `aug1`. `aug2` et `aug3` ne font pas partie de cette nouvelle passe.

| Partition | Vues acceptées |
|---|---:|
| Entraînement | 1 841 540 |
| Validation complète préparée | 99 318 |
| Test préparé | 97 432 |
| Total accepté | 2 038 290 |

Il s’agit de **vues**, comprenant des variantes augmentées, et non de 2 millions de documents indépendants. Les 660 rejets historiques sont comptabilisés. Les variantes d’un original héritent de sa partition. La préparation conserve les textes, diacritiques et alphabets de chiffres ; les cellules de tableaux sont réinsérées à leur position.

Preuve : [corpus-completion.json](../../results/training/markdown-v3/corpus-completion.json). Elle comptabilise 534 066 pages avec tableaux, 687 512 tableaux et 15 624 126 cellules sources. Le plus long échantillon traité contient 3 587 tokens sous cette configuration.

Markdown ne restitue pas nativement toute la géométrie des cellules fusionnées. Le contenu est conservé à l’ancre et les positions couvertes sont laissées vides ; les spans sont audités. Le CER/WER n’évalue pas à lui seul cette géométrie.

## Passe principale effectivement exécutée

| Paramètre | Valeur |
|---|---|
| Initialisation des adaptateurs | Parent 14355 vérifié ; optimiseur et curseur neufs pour la nouvelle passe |
| GPU / parallélisme | 1 H200, WORLD_SIZE=1 |
| Microbatch / accumulation / batch global | 1 / 384 / 384 ; dernier lot partiel |
| Couverture | Une passe sur 1 841 540 vues |
| Optimiseur | AdamW, weight decay 0,01, gradient clip 1 |
| Learning rate | Warmup 100 mises à jour vers 2e-5, puis cosine vers 1e-5 |
| Monitoring | 128 pages avec tableaux + 128 pages contrôle |
| Fin | Phase 4796, compteur historique global 19151 |
| Dernière train loss | 0,1354276337 |
| Loss de validation finale | 0,1596224839 |
| Validation tableaux / contrôles | 0,1901880494 / 0,0751162943 |

La validation de monitoring porte sur **256 pages**, pas sur toutes les 99 318 vues préparées. La loss globale est pondérée par les tokens supervisés. La train loss est celle de la mise à jour enregistrée ; les deux losses ne sont pas un même estimateur.

L’initiale de validation de cette passe est 0,4066384454 : elle correspond au **parent déjà adapté**, pas au Qwen vierge. La validation finale baisse, mais elle remonte légèrement par rapport au point 4707 (0,1588523771). Cette seule variation ne permet pas d’identifier une cause de surapprentissage.

Le code réellement embarqué est conservé dans [experiments/markdown-v3/code](../../experiments/markdown-v3/code). Ses empreintes correspondent à la configuration enregistrée. Les fichiers dans `scripts/` comprennent aussi des versions de préparation, d’expérimentation et des variantes historiques ; ne pas déduire d’un nom de script qu’il a été déployé.

## Arrêt et reprise

L’exécution s’arrête au pas 4682, sous la réserve disque de sécurité. Le diagnostic d’arrêt rencontre aussi un dossier `ocr-stopped` manquant. Le 26 octobre, le PVC est étendu de 128 à 160 Gio et le dossier est créé. La reprise recharge le même état vérifié (adaptateurs, AdamW, RNG et curseur), contrôle un replay réel puis reprend à 4683. Les sources participant au hash restent inchangées.

Les logs conservés couvrent **toutes les 4 796 mises à jour consécutives** et **202 événements de validation**. Voir [l’index des logs](../../results/logs/inventory.json). La vérification indépendante finale contrôle la couverture, les tenseurs et le SHA du checkpoint sur les deux stockages.

Checkpoint final 4796 : `0d9cdd97a3d30bdf432ae198784c9ca55d071d3c269f118d75bf6c55bf211d92`.

## Essai correctif indépendant du 28 octobre

Il démarre avec une **LoRA neuve sur le Qwen original**, sans hériter de l’adaptateur en régression. Corpus : 375 exemples, 300 entraînement / 75 validation, partitions par groupe documentaire ; 299 pages publiques, 61 contrôles littéraux et 15 contrôles blancs. Les 31 documents du benchmark historique sont exclus. Les annotations publiques ne sont pas toutes retranscrites indépendamment par une personne.

Configuration : 2 époques, batch 16, 38 mises à jour, 600 exemples consommés, LoRA 16/32, contexte 8192, LR 1e-5 vers 2e-6. Validation 0,6415951588 → 0,5635696377, soit −12,16 %. Ce progrès de loss ne suffit pas : erreurs de chiffres persans/arabo-indiens et de signes persistantes. **Aucune promotion automatique.**

Adaptateur epoch-2 : `f6f4c76626a747d0bcd1cd047f2c8d43bb6b3091904f9377c85bb235754307e0`. [Configuration, événements, générations et preuve de fin](../../results/training/corrective-v1) ; [bilan](../../reports/corrective/BILAN.md).
