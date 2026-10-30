# Essai correctif indépendant — bilan du 28 octobre 2026

La phase bornée a terminé ses **38 mises à jour et 600 exemples consommés**
(300 pages d’entraînement parcourues deux fois, 75 pages de validation).
Un nouvel adaptateur LoRA a été entraîné depuis le Qwen de base vérifié.
Les checkpoints historiques 4707 et 4796 sont conservés.

La loss de validation globale passe de **0,641595 à 0,563570**, soit une baisse
relative de **12,16 %**. Les trois langues baissent : arabe 1,386889 → 1,209046 ;
français 0,129285 → 0,119105 ; anglais 0,119364 → 0,112122.
Le graphe des valeurs mesurées est dans `losses-correctives.png`.

Les 21 sorties de contrôle sont récupérées : 6 après la première époque et 15
après la seconde. Les références ne sont pas données au modèle pendant la génération.
Les données et sorties exactes sont dans `mesures.json` et `../retrieved-corrective/`.

**Le candidat n’est pas validé pour remplacement du modèle historique.**
Le contrôle français isolé conserve exactement les signes testés. En revanche,
des confusions entre chiffres arabo-indiens et persans subsistent sur la sonde
de chiffres. Le contrôle arabe de validation atteint la limite de sortie et perd
encore des chiffres et des symboles ; le contrôle anglais avec chiffres persans
contient aussi des erreurs.
Une baisse de loss et un arrêt EOS ne prouvent donc pas la suppression des
hallucinations ou la fidélité de tous les caractères.

Les pages blanches et les autres sorties doivent être lues avec les limites de
leur contrôle : la page parfaitement blanche est filtrée avant génération, les
pages de texture papier sont réellement présentées au modèle. Les annotations
des documents publics n’ont pas toutes été retranscrites indépendamment.

Adaptateur de fin d’époque 2, SHA-256 :
`f6f4c76626a747d0bcd1cd047f2c8d43bb6b3091904f9377c85bb235754307e0`.
Aucun déploiement automatique ni remplacement des résultats du comparatif.
Le comparatif de quatre modèles conserve la mesure historique du checkpoint 4707.
