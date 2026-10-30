# Points à discuter avec l’encadrant

## 1. Définir une validation métier figée

Constituer un corpus arabe/français/anglais représentatif des documents de production, avec transcription vérifiée humainement et documents disjoints de l’entraînement. Conserver des sous-groupes : tableaux, plusieurs colonnes, petites polices, chiffres arabes/persans/latins, signes spéciaux, pages presque blanches et manuscrits contemporains. Fixer la méthode d’annotation des zones non transcrites.

**Décision attendue :** familles documentaires prioritaires, volume, annotateurs et budget. Le benchmark public exploratoire actuel sert à détecter les régressions, pas à homologuer seul la production.

## 2. Distinguer amélioration de loss et fidélité de génération

Les signes `œ/æ/ç/É/À` ont montré une régression sur des sondes du modèle final, alors que la base les copiait. Les répétitions et confusions de chiffres persistent dans certains essais. Les contrôles avec/sans cache et le retour à la résolution native ont permis d’isoler des problèmes de protocole, sans expliquer tous les défauts.

**Décision attendue :** audit ciblé des paires image/cible et de la distribution des langues, puis ablations une variable à la fois. Ne pas présumer que davantage d’époques éliminera les hallucinations. Le petit candidat correctif montre une piste, pas une correction acquise.

## 3. Comparer correctement les checkpoints

Les scores publiés de notre modèle sont ceux de **4707** ; la fin technique est **4796** ; le candidat correctif repart de la **base**, avec une **LoRA neuve**. Ce sont trois modèles à distinguer.

**Décision attendue :** évaluer de façon appariée base, 4707, 4796 et candidat sur le même corpus gelé, avec les mêmes images et paramètres. Conserver toutes les sorties, les raisons d’arrêt et les cas extrêmes. Choisir un checkpoint selon validation documentée, pas selon le seul dernier pas.

## 4. Fixer les critères d’acceptation avant un nouveau lancement

Proposition à valider : CER/WER par langue et layout, absence de régression sur les contrôles littéraux, conservation des alphabets de chiffres, taux de sortie interrompue, omissions après la fin de page, inspection des tableaux, latence bout en bout et consommation mémoire. Définir séparément un signal de répétition et une hallucination confirmée par lecture humaine.

**Décision attendue :** seuils, règle de rejet/retour à un autre OCR et degré de validation humaine. Un garde-fou de génération peut bloquer une boucle, mais ne transforme pas une transcription erronée en texte exact.

## 5. Choix de service et de déploiement

Unlimited est exécuté localement sur le cluster ; Mistral est un service externe. Les résultats actuels donnent une meilleure moyenne à Mistral sur les deux corpus comparés, avec des limites fortes sur l’arabe historique. Une décision doit aussi traiter confidentialité, conditions de réutilisation, coûts, débit et disponibilité.

**Décision attendue :** priorité à la souveraineté d’hébergement, au niveau d’erreur ou à une combinaison avec validation humaine. Les timings actuels ne comparent pas deux matériels identiques. Le manuscrit arabe moderne nécessite un jeu propre avant toute promesse métier.

## Lecture proposée pour la réunion

1. Cinq minutes : vue d’ensemble et tableau des états.
2. Dix minutes : résultats par langue, puis deux bonnes et deux mauvaises sorties.
3. Dix minutes : courbes de loss, provenance des données et diagnostic de fidélité.
4. Fin de réunion : choisir une prochaine expérience bornée, ses métriques et ses conditions de promotion.
