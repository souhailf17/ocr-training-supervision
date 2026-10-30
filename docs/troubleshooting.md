> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

> Historical scheduling note: a replacement probe later succeeded with scheduler-selected placement. The issue list below is a discovery log; do not treat it as the current job state.

> The issue list below is a historical discovery log, not live status. The 2 October 2026 saved run configuration and verified continuation are documented in [`training.md`](training.md).

# Troubleshooting

- SSH banner timeout on the access host: resolved on retry; do not restart remote services.
- An alternate access host had no configured kubeconfig; use the approved project access host.
- Hugging Face HTTP 401: use the access host's approved credential; public metadata alone is insufficient.
- PowerShell TLS failure: Python urllib worked with certificate verification enabled.
- RBAC forbidden for nodes/all namespaces/PriorityClasses: obtain read access or the approved class name; do not escalate privileges.
- Probe admission rejects Never: use a PriorityClass configured for Never; do not remove the protection merely to make scheduling succeed.
- Transfer mismatch: keep the failure log, retry the current source object, and never report it as verified.
- Pending GPU: wait for genuinely available resources; do not delete or scale other workloads.

The initial failed runtime probe remains suspended for inspection. Only resources
labelled project=q35ocr-20260921-p560 were created or changed by this task.
