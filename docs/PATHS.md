# Pfade und Mounts

lora-lab läuft als eigener Container und rendert nicht selbst — ComfyUI macht
das auf der GPU. Der Container braucht deshalb nur **lesenden Zugriff** auf
Bilder und **schreibenden** auf seine eigenen Daten.

## Mount-Vertrag

Feste Pfade im Container, damit die Config unabhängig von deinem Unraid-Layout
bleibt:

| Host (Unraid) | Container | Modus | Wofür |
|---|---|---|---|
| `/mnt/user/appdata/lora-lab/config` | `/app/config` | rw | Konfiguration inkl. `local/` |
| `/mnt/user/appdata/lora-lab/data` | `/app/data` | rw | SQLite, gerenderte Bilder, Reports |
| `/mnt/user/Datasets` | `/datasets` | ro | Referenz- und Trainingsordner |
| `/mnt/user/appdata/comfyui/output` | `/comfy-output` | ro | fertige Bilder (schneller als HTTP) |
| `/mnt/user/appdata/comfyui/models` | `/models` | ro | optional: LoRA-Dateien für Metadaten/Hash |
| `/mnt/user/appdata/fizgig/user_data` | `/fizgig` | ro | optional: Run-Metadaten, Loss-Logs |

Nur die ersten drei sind Pflicht. Ohne `/comfy-output` werden die Bilder über
`/view` heruntergeladen — funktioniert, ist aber langsamer.

## Was in die Config kommt

In `config/local/subjects.yaml` stehen **Container-Pfade**, nicht Host-Pfade:

```yaml
subjects:
  - id: subject_a
    trigger: subja
    ref_dir: /datasets/subject_a/refs      # held-out, nicht im Training
    train_dir: /datasets/subject_a/train   # das Trainingsset
```

In `config/local/project.yaml`:

```yaml
comfyui_url: http://comfyui:8188     # gleicher Docker-Netzwerkname
comfy_output_dir: /comfy-output
```

Die LoRA-Dateien in `grid.yaml` bleiben **ComfyUI-relative Namen**
(`moss/...`), keine Container-Pfade — ComfyUI lädt sie, nicht lora-lab.

## Referenzordner

Der Identitäts-Score ist nur aussagekräftig, wenn die Referenzen nicht im
Training waren. Empfehlung pro Subjekt:

- 15–30 Bilder, Gesicht klar erkennbar
- verschiedene Winkel, Lichtsituationen und Brennweiten
- keine Bilder aus dem Trainingsset, keine Duplikate
- ein Ordner pro Subjekt, flach oder verschachtelt (wird rekursiv gelesen)

Für die Kalibrierung werden mindestens **zwei Subjekte** mit Referenzen
gebraucht: eines zum Matchen, eines für die Impostor-Verteilung. Mit nur einem
Subjekt lässt sich keine sinnvolle Schwelle fitted — dann bricht `score
--identity` mit einer klaren Meldung ab.

## Prüfen

```bash
lora-lab doctor        # Pfade, Bildzahlen, offensichtliche Überlappung
lora-lab check-refs    # exakte Duplikate Referenz <-> Training (SHA256)
lora-lab audit /datasets/subject_a/train
```

`check-refs` gibt `0` zurück, wenn alles sauber ist, und `1` bei einem Leak —
damit taugt es als Pre-flight vor einem langen Lauf.

## Wenn du mir die Pfade gibst

Am einfachsten als YAML-Block im Chat:

```yaml
datasets_root: /mnt/user/Datasets
comfy_output: /mnt/user/appdata/comfyui/output
fizgig_user_data: /mnt/user/appdata/fizgig/user_data   # optional
subjects:
  - id: ...
    refs: <host-pfad oder "none">
    train: <host-pfad>
```

Ich schreibe daraus `config/local/subjects.yaml`, `config/local/grid.yaml` und
den Compose-Block. Diese Dateien sind per `.gitignore` ausgeschlossen und
landen nicht im öffentlichen Repo.
