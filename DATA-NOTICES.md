# Sources et conditions de réutilisation

Les licences des données restent distinctes de celles des modèles et du code. Ce dépôt ne concède pas une nouvelle licence générale sur des données tierces. Les métadonnées et références sources sont conservées dans `results/benchmarks/*/jeu.json`.

| Source | Usage dans ce projet | Notice |
|---|---|---|
| [OCR-Data/ocr_data](https://huggingface.co/datasets/OCR-Data/ocr_data) @ `4cf05556e21bc54aca54a8b1f81ce172f3ef0477` | Corpus principal originaux + augmentations | Corpus massif et poids non redistribués ; vérifier les droits des documents sources avant autre usage |
| [NayanaOCRBench Natural](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCRBench_Natural) @ `2684870ad1a0ba271ed56c31e19cb05a6b78c291` | 31 imprimés et une partie du correctif | CC BY-NC 4.0 : attribution et restriction non commerciale |
| [RASAM 1](https://github.com/calfa-co/rasam-dataset) / BULAC | 15 scans manuscrits arabes historiques | Annotations Apache 2.0 ; droits des images des collections BULAC et attributions à conserver ; ne pas confondre licence des annotations et droits des scans |
| [RIMES](https://zenodo.org/records/10812725) | 8 lettres manuscrites françaises | CC BY-NC 4.0 ; signatures non transcrites masquées seulement dans les entrées communes |
| [Revolutionary City](https://zenodo.org/records/15776323) | 8 images manuscrites anglaises historiques | CC BY 4.0 ; sources APS/HSP indiquées par page ; conserver les attributions des collections |
| [NOD / Hegghammer](https://doi.org/10.5281/zenodo.5068735) | Pages publiques du corpus correctif | CC BY 4.0 ; Thomas Hegghammer (2021) |

Les références publiées ne sont pas toutes retranscrites indépendamment par l’équipe. Les annotations ambiguës ou incomplètes identifiées avant inférence sont exclues ou leurs zones non transcrites traitées selon une politique documentée. Une anomalie imprimée découverte ensuite reste dans l’analyse principale et fait l’objet d’une sensibilité séparée.

Les exemples des rapports et les textes des résultats sont diffusés pour l’étude et la discussion du projet, avec ces restrictions. Ils ne constituent pas un corpus libre de contraintes pour un usage commercial. Aucun document client, token API, clé SSH, secret S3 ou poids de modèle n’est ajouté à ce dépôt.
