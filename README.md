# Worker RunPod : prep

Modèle : `org/model`. Dépôt : https://github.com/romingodelfuego/fakinflu-prep-worker, monté en sous-module dans fakinflu.
**La source fait foi ici** : les changements passent par des PR sur ce dépôt.

## Cycle de vie (depuis fakinflu)

```bash
make worker-status W=prep                  # sous-module vs main vs dernière release (0 crédit)
make worker-bump W=prep [BUMP=minor] PR=1  # VERSION + worker.json dans la PR
# merge de la PR sur main = build RunPod (chaque push sur main reconstruit l'image)
make worker-release W=prep NOTES="..."     # tag vX.Y.Z + GitHub Release, puis git add du sous-module
```

**Première release seulement** : il faut lier le dépôt dans la console RunPod
(**Serverless → New Endpoint → Import Git Repository**). Appliquer ensuite les réglages
de `worker.json` (GPU, workers, timeouts) et noter l'ID dans `.env` → `RUNPOD_PREP_ENDPOINT_ID`.
Claude peut faire tout ça à ta place (skill `deploy-worker`).

## Contrat d'API

```json
{"input": {"items": [{"id": "a", "...": "..."}]}}
→ {"results": [{"id": "a", "...": "..."}, {"id": "b", "error": "..."}]}
```

Test local du handler (sur une machine GPU) : `python handler.py --test_input "$(cat test_input.json)"`.
