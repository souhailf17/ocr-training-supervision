> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Runtime environment

The 2 October 2026 training snapshot recorded Python 3.13, PyTorch 2.11.0+cu130, Transformers 5.8.0 and Forge Arsenal/CLI 2026.7.11. The optimized GPU environment used `fla-core==0.5.2`, `causal-conv1d==1.7.0` and Triton `3.7.1` on NVIDIA H200 hardware.

These package versions describe the recorded runtime, not a newly verified live environment. The repository does not vendor Forge Arsenal, its scheduler extension, the model or dataset. In particular, `lr_scheduler_basic` is provided by the authorized runtime and its source is not included in this snapshot. A normal `pip install -r requirements.txt` alone is not sufficient to reproduce the training job.

Do not put cluster hostnames, OpenShift tokens, kubeconfigs or object-store credentials in source files. Supply credentials at runtime through an OpenShift Secret. The source launch scripts reference the original project layout and namespace; review them before use in another environment.
