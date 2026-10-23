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
| `/mnt/user/ai/training/datasets` | `/datasets` | ro | Referenz- und Trainingsordner |
| `/mnt/user/ai/models` | `/models` | ro | LoRA-Dateien, Metadaten |
| `/mnt/user/ai/training/fizgig/Fizgig/user_data` | `/fizgig` | ro | Run-History, Loss-Logs |
| `/mnt/user/ai/output` | `/comfy-output` | ro | fertige Bilder (schneller als HTTP) |

ComfyUI läuft unter `http://192.168.188.62:8188` (oder über Tailscale); der
Container erreicht es per LAN-IP.

Config und Data sind Pflicht, `/datasets` für Referenzen. Ohne `/comfy-output`
werden die Bilder über `/view` heruntergeladen — funktioniert, ist aber
langsamer.

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

## Config-Dateien

Beim ersten Start legt der Entrypoint fehlende Dateien aus `/app/defaults` in
`/app/config` ab: `project.yaml`, `subjects.yaml`, `scenarios.yaml`,
`grid.yaml`. Vorhandene Dateien werden nie überschrieben.

`discover --apply` schreibt danach:

| Datei | Inhalt |
|---|---|
| `config/local/subjects.yaml` | erkannte Subjekte mit Container-Pfaden |
| `config/local/loras.yaml` | erkannte LoRAs mit Epoche und Rank |
| `config/local/grid.yaml` | Kopie des Grids, mit `loras_file` statt inline-LoRAs |

Die Datei in `config/local/` gewinnt immer gegen die gleichnamige Datei in
`config/`.

## Automatisch statt abtippen

Der Web-Port ist über die Stack-Variable `LORA_LAB_PORT` konfigurierbar
(Default 8199); im Container bleibt er fest auf 8099.

Im Container:

```bash
docker exec -it lora-lab lora-lab --root /app discover \
  --datasets /datasets --loras /models/loras --apply
```

Das schreibt `config/local/subjects.yaml` und `config/local/loras.yaml`;
danach in `config/local/grid.yaml` einmal `loras_file: config/local/loras.yaml`
setzen (oder die inline `loras` entfernen). Erkannt werden `refs`/`reference`/
`heldout` und `train`/`images`/`dataset` als Unterordner, Trigger aus
`trigger.txt` oder den `ss_*`-Metadaten der LoRA-Datei.

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
