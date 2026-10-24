# Changelog

Nennenswerte Änderungen an lora-lab. Format nach
[Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach
[SemVer](https://semver.org/lang/de/). Ältere Änderungen stehen in der
Git-Historie.

## [Unreleased]

### Fixed

- Run-/LoRA-Löschen zerstört keine Dateien mehr, die ein anderer Run über wiederverwendete Jobs noch referenziert. Solche Dateien bleiben auf der Platte, bis der letzte verweisende Run gelöscht ist.
- `generate` nutzt jetzt das live abgefragte `/object_info` für die UI→API-Konvertierung statt immer auf die eingebaute Fallback-Tabelle zu fallen. Workflows mit Nodes außerhalb der Tabelle werden damit korrekt gemappt.
- `--verdicts` meldet jetzt auch einen Konflikt mit einem in `project.yaml` konfigurierten VLM statt ihn still zu ignorieren.
- Der Ordner-Browser bricht bei 300 Ordnern nicht mehr mitten im Zählen der Bilder ab.
- `/images/{path}` lehnt Pfade außerhalb des Bildordners korrekt ab (`is_relative_to` statt `startswith`).
- Referenz-Upload endet nicht mehr mit „connection was reset“. Der
  Multipart-Body wird gelesen, bevor das Dateisystem angefasst wird; Fehler
  kommen als Meldung im UI an.
- `/datasets/_refs` ist im Compose-File zusätzlich als `:rw` gemountet. Der
  Default-Referenzordner lag zuvor in einem read-only Mount, weshalb der erste
  Upload immer mit `FileNotFoundError` auf `mkdir` scheiterte.
- Nicht beschreibbare Referenzordner werden vor dem Schreiben erkannt und mit
  Hinweis auf den Mount gemeldet, statt eine Exception zu werfen.
- Ungültige Uploads (keine Bilddatei, zu groß, leer) liefern eine Meldung
  statt eines 500ers.

### Added

- Referenzen-Seite warnt, wenn der Ordner nicht beschreibbar ist.
- `lora-lab doctor` prüft für jedes Subjekt den effektiven Referenzordner —
  auch den Default für Subjekte ohne `ref_dir` — und warnt, wenn Uploads
  scheitern würden.
- `score` bewertet Bilder parallel (CLI: `--workers`, Standard 4; Web-UI: 4).
  Die VLM-Anfragen laufen gleichzeitig; die ONNX-Gesichtserkennung bleibt
  serialisiert und braucht dadurch keinen zusätzlichen RAM. Flüchtige
  VLM-Fehler (Verbindungsabbruch, 5xx) werden einmal wiederholt.

### Changed

- Der Scorer-Aufbau (Health, Adherence, Identity) liegt einmal in
  `scorers/run.py` und wird von CLI und Web-UI geteilt statt doppelt gepflegt.
- SQLite committet nicht mehr pro Zeile, sondern pro Job bzw. pro
  Scoring-Batch — deutlich weniger fsyncs auf gemounteten Volumes.
- Der Web-Report wird pro Run gecacht und nur bei geänderten
  Jobs/Bildern/Scores neu berechnet.
- Toter Code entfernt (`estimate_seconds`, `queue_remaining`, ungenutzter
  `object_info`-Parameter an `build_prompt`).
