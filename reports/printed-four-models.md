# Conclusion du comparatif — 31 images, quatre modèles

Les quatre modèles ont été exécutés sur les mêmes 15 images arabes, 8 françaises et 8 anglaises. Les 93 résultats antérieurs ont été conservés et leurs scores ont été reproduits à l’identique. Unlimited-OCR-2 ajoute 31 nouvelles transcriptions réelles. Aucun texte de référence n’a été donné au modèle pendant la génération.

| Modèle | CER global ↓ | WER global ↓ | CER médian par page ↓ |
|---|---:|---:|---:|
| Notre OCR — checkpoint 4707 | 95,18 % | 109,96 % | 23,60 % |
| Qwen3.5-9B de base | 61,59 % | 72,87 % | 12,89 % |
| dots.mocr | 571,22 % | 19,25 % | 6,17 % |
| Unlimited-OCR-2 | 7,48 % | 14,75 % | 3,60 % |

Le CER global le plus faible est obtenu par **Unlimited-OCR-2**. Le WER global le plus faible est obtenu par **Unlimited-OCR-2**. La meilleure médiane du CER par page revient à **Unlimited-OCR-2**.

## Comparaisons page par page

- Face à Notre OCR — checkpoint 4707, Unlimited-OCR-2 a un CER inférieur sur 23 pages, égal sur 0 et supérieur sur 8.
- Face à Qwen3.5-9B de base, Unlimited-OCR-2 a un CER inférieur sur 22 pages, égal sur 0 et supérieur sur 9.
- Face à dots.mocr, Unlimited-OCR-2 a un CER inférieur sur 17 pages, égal sur 1 et supérieur sur 13.

## Résultats par langue

| Langue | CER minimal | WER minimal | CER médian minimal |
|---|---|---|---|
| Arabe | Unlimited-OCR-2 | Unlimited-OCR-2 | Unlimited-OCR-2 |
| Français | Unlimited-OCR-2 | Unlimited-OCR-2 | Unlimited-OCR-2 |
| Anglais | Unlimited-OCR-2 | Unlimited-OCR-2 | Unlimited-OCR-2 |

## Choix proposé

Pour ce jeu précis et ces réglages, **Unlimited-OCR-2 est le meilleur choix global** : il arrive premier sur les trois indicateurs ci-dessus. Il faut néanmoins vérifier les pages difficiles avant un usage métier.

Même ce modèle garde des erreurs : fr-02 (CER 69,66 %), ar-08 (CER 28,60 %), fr-05 (CER 25,01 %). Les comparaisons page par page ci-dessus montrent aussi les cas où un autre modèle fait mieux.

## Limites de la décision

Notre OCR est la copie historique du **checkpoint 4707**, après 1 807 488 exemples. Ce tableau ne mesure pas le checkpoint final 4796 ni la version corrective en préparation.

Le CER global additionne les erreurs sur toutes les pages ; une très longue répétition peut fortement le dégrader. Le score global élevé de dots.mocr doit donc être lu avec sa médiane et les sorties par page. Aucune réponse répétitive ou interrompue n’a été retirée pour améliorer les scores. Un arrêt normal ne garantit pas une transcription exacte.

Les coordonnées et balises PAGE/BLOCK/FORK/LEVEL/MERGE d’Unlimited-OCR-2 sont des instructions de structure, pas du texte imprimé. Le décodeur les retire sans réécrire le contenu des blocs. En-têtes, pieds de page, numéros, signes et passages incertains restent inclus. Les images annotées montrent les boîtes prédites, pas des boîtes validées par un annotateur.

Les taux reposent sur les mêmes fonctions de retrait du balisage HTML/Markdown, de normalisation Unicode NFC et des espaces. Les familles de chiffres, accents, diacritiques et ponctuations restent significatifs. Les références éditeur n’ont pas été retranscrites intégralement de façon indépendante, et le chevauchement avec les préentraînements est inconnu. Ces 31 pages constituent un essai illustratif, pas une preuve de supériorité universelle.

Les modèles utilisent leurs propres consignes/processeurs. Le test Unlimited utilise le moteur Transformers natif BF16, 6 000 tokens maximum, une fenêtre de 8 192 tokens et une limite de 300 secondes par image. La révision des poids est 5632329a32e103d5f1add270681e0f5c1233c6e4 ; les fichiers ont été vérifiés par SHA-256. Les images n’ont pas été envoyées à la démonstration publique.

Les poids des modèles ne sont pas inclus dans ce dossier. Images et références Nayana : CC BY-NC 4.0. Source du modèle : https://huggingface.co/baidu/Unlimited-OCR-2 .
