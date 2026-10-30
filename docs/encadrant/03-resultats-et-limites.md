# Résultats mesurés et limites

## Trois expériences distinctes

Toutes utilisent 15 images arabes, 8 françaises et 8 anglaises. Le corpus manuscrit est distinct du corpus imprimé. Les mêmes sorties Unlimited imprimées sont réutilisées dans les deux études imprimées : ce ne sont pas 31 nouvelles exécutions indépendantes.

### Imprimés : quatre modèles, checkpoint intermédiaire

| Modèle | CER micro % | WER micro % | CER médian par image % |
|---|---:|---:|---:|
| Notre Qwen adapté, checkpoint **4707** | 95,17555 | 109,96346 | 23,60248 |
| Qwen3.5-9B original | 61,59280 | 72,87188 | 12,89063 |
| dots.mocr | 571,22425 | 19,24979 | 6,17284 |
| Unlimited OCR 2 | **7,48260** | **14,74635** | **3,59833** |

[Sorties, références et scores par image](../../results/benchmarks/printed-four-models/resultats.json). Les 124 scores sont conservés, y compris les mauvaises sorties. Notre adaptation gagne contre la base sur 12 images et perd sur 19. Elle a un CER arabe plus faible que la base dans ce jeu, mais son score global est moins bon.

Le très grand CER de dots.mocr provient notamment d’une répétition de ponctuation sur `fr-07` (324 981 caractères). Le découpage en mots explique qu’une telle répétition affecte différemment CER et WER. Ne pas masquer cette sortie ni transformer sa médiane en performance globale.

Le protocole initial à 1 120 pixels a été invalidé après contrôle de résolution. Le tableau publié retient le protocole corrigé à 1 024 pixels, correspondant à l’entraînement, et des processus Qwen séparés avant/après avec audit de reproduction. Les scores 1 120 pixels ne sont pas mélangés à cette conclusion.

### Imprimés : Mistral et Unlimited

| Modèle | CER micro % | WER micro % |
|---|---:|---:|
| Mistral OCR 4.1 | **5,89713** | **11,56492** |
| Unlimited OCR 2 | 7,48260 | 14,74635 |

[Rapport scientifique](../../reports/printed/rapport-scientifique.md), [PDF](../../reports/printed/rapport-scientifique.pdf), [62 résultats](../../results/benchmarks/printed-mistral-unlimited/resultats.json).

Une référence imprimée `fr-02` comporte une duplication confirmée ; le score principal reste intact et une analyse de sensibilité sans cette image est fournie dans le rapport. Le bootstrap apparié tient compte des groupes documentaires, avec seulement 23 groupes. Il ne prouve pas une supériorité sur tous les documents métier.

### Manuscrits réels : Mistral et Unlimited

| Population | n | CER Mistral % | CER Unlimited % | WER Mistral % | WER Unlimited % |
|---|---:|---:|---:|---:|---:|
| Ensemble | 31 | **56,57** | 250,39 | **75,18** | 340,50 |
| Arabe | 15 | 79,91 | 336,75 | 98,21 | 429,00 |
| Français | 8 | 3,86 | 7,65 | 14,48 | 21,72 |
| Anglais | 8 | 10,16 | 114,84 | 26,27 | 207,92 |

[Rapport](../../reports/handwriting/rapport-manuscrit.md), [PDF](../../reports/handwriting/rapport-manuscrit.pdf), [62 résultats](../../results/benchmarks/handwritten-mistral-unlimited/resultats.json).

Sources : RASAM/BULAC (arabe historique maghrébin, 3 manuscrits), RIMES (lettres françaises modernes), Revolutionary City (anglais du XVIIIe siècle). Les layouts sont réels et variés ; aucune police manuscrite synthétique n’est utilisée. Les signatures françaises sans transcription sont masquées identiquement pour les deux modèles. Cinq images arabes sources ne font qu’environ 724 pixels de haut ; certaines images anglaises contiennent deux pages physiques. Les références sont les annotations publiées, sans retranscription indépendante exhaustive.

Unlimited atteint le budget de génération sur 10 images ; toutes restent incluses. La dérive de `en-08`, à référence très courte, pèse lourd dans le score anglais. Les deux modèles restent insuffisants sur ce corpus arabe historique. Cela ne tranche pas la qualité de l’arabe manuscrit contemporain.

## Comment interpréter les métriques

`CER = 100 × distance de Levenshtein en caractères / caractères de référence` ; `WER` applique la même formule aux mots. La micro moyenne additionne erreurs et longueurs avant division. La médiane décrit une page typique, pas l’impact total des sorties extrêmes.

Le nettoyage visible HTML/Markdown, NFC et regroupement des espaces est identique entre référence et prédiction. Casse, ponctuation, diacritiques et alphabets des chiffres restent distincts. Seul le marqueur éditorial anglais `^` du corpus manuscrit est retiré symétriquement. Les références ne sont jamais fournies lors de l’inférence.

**Un CER/WER supérieur à 100 % est possible** quand les insertions sont nombreuses. `100 − CER` n’est pas une accuracy. Ces mesures ne quantifient pas toute la structure des tableaux, le positionnement, ni une hallucination sémantique. EOS ou HTTP 200 ne prouvent pas une transcription complète.

## Conclusion utilisable aujourd’hui

Sur les imprimés testés, Mistral a le meilleur score moyen dans le face-à-face et Unlimited constitue une option locale nettement meilleure que nos sorties Qwen 4707 dans le comparatif historique. Sur le manuscrit français moderne testé, Mistral est également meilleur en moyenne. L’arabe historique nécessite un travail spécifique.

Il n’existe pas ici de vainqueur universel : corpus réduit, éventuel recouvrement avec les données préalables des modèles inconnu, coût et latence non contrôlés, références imparfaites. Les temps de l’API Mistral et de l’inférence GPU Unlimited ne sont pas un benchmark matériel équivalent. La décision de production doit intégrer confidentialité, hébergement, fidélité et détection de dérives.

**Le candidat correctif et le checkpoint final 4796 n’ont pas remplacé la colonne 4707 dans le comparatif.** Une réévaluation appariée de ces candidats est une expérience à prévoir, pas un résultat disponible.
