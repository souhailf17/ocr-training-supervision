Reprise manuelle demandée le 26 octobre 2026
==========================================

Le job `q35ocr-20260921-p560-markdown-gpu1-w9-r4682-v2`, UID
`9508ac42-8510-4397-9726-48c90774d45a`, poursuit le même corpus, dans le même
répertoire et avec le même préfixe MinIO. La surveillance de l’application reste
en pause. Aucun mail de réussite n’a été envoyé.

Checkpoint de départ : `checkpoint-stopped-004682.pt`, 1 797 888 exemples,
SHA-256 `db14c58ed66ca6ffc4cbc140d03da9f317ec89166ea94dfbebab4472fae18cbe`.
La totalité du fichier sur PVC et MinIO, les états AdamW/RNG, le curseur exact
et les 206 archives ont été contrôlés avant reprise. Aucun code participant au
hash de configuration n’a été changé. Le précontrôle réel de reprise a produit
une erreur de poids et de loss égale à zéro, puis a restauré le point initial.

Le PVC était sous le seuil de sécurité, avec 3 914 694 656 octets libres.
Extension prise en charge par la classe de stockage : 128 vers 160 Gio.
37 735 464 960 octets libres après extension, aucun fichier supprimé.
Le répertoire `ocr-stopped`, absent et responsable de l’échec secondaire du
diagnostic précédent, a été créé avant la reprise. Le lancement garde un verrou
et un état distinct `resume-20261026-execution.json`, sans confondre l’ancien
fichier d’erreur avec la nouvelle exécution.

Paramètres conservés : un H200 sur worker9, WORLD_SIZE=1, microbatch 1,
accumulation 384, batch global 384, contexte 8192, LoRA 16/32. Le dernier lot
partiel utilise naturellement moins de 384 exemples. Le job historique à quatre
GPU demeure suspendu et aucun autre workload n’a été modifié.

`resume-confirmed.json` contient le dernier état observé. Il ne vaut pas preuve
de fin : le calcul était encore en cours au moment de son écriture.

Le job dépendant `q35ocr-20260921-p560-md-final-verify-r4682` attend la sortie de
cette exécution puis contrôle, sans GPU et avec PVC en lecture seule : totalité
des exemples, curseur final, corpus complet, précontrôles initial et de reprise,
configuration, tenseurs finis, SHA complet du checkpoint sur PVC et MinIO, losses
réellement enregistrées et 12 mêmes sorties OCR avant/après. Il ne notifie et
n’envoie aucun mail. Son résultat figure dans ses logs Kubernetes sous le marqueur
`TRAINING_COMPLETION_VERIFIED` ; aucune qualité OCR universelle n’est présumée.

Pour une prochaine lecture manuelle, exécuter `read_resume_status.py` ou
`read_resumed_completion.py` avec le pont SSH autorisé `remote_safe.py`.
Le contrôle historique `markdown-corpus-20261013/check_completion.py` vise encore
l’ancien job arrêté : ne pas l’utiliser pour déduire l’état de cette reprise.
