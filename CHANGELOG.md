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

- **Gold-Set & Judge-Kalibrierung:** Seite `/label` zum Labeln von Attributen und Defekten ohne Anzeige der VLM-Antwort. Die Übereinstimmung des Judges wird als Cohens κ gemessen (`lora-lab calibrate`, Report-Hinweis). Ab 30 Antworten fliegt ein Judge mit κ < 0,4 aus dem Ranking.
- **Defekt-Judge:** `DefectScorer` fragt den VLM nach den bildlich prüfbaren Defekten aus dem Katalog (neue Achse `defects`). Kritische Defekte machen das Bild zum Ausfall, Schweregrad ≥ 2 verhindert einen Treffer. Das Ranking zeigt eine Defektrate, die Bildansicht listet die gefundenen Defekte. Kostet eine zweite VLM-Anfrage pro Bild.
- **Blind-A/B-Abnahme:** Seite `/duel` vergleicht zwei anonyme Bilder mit gleichem Subjekt, Szenario und Seed aus den Top 6 des Rankings und stellt Paare mit den wenigsten Stimmen zuerst. Der Stand wird per Bradley-Terry mit Bootstrap-Intervall berechnet.
- **Gepaarter Vergleich:** `lora-lab compare A B [--run-a --run-b]` und ein Abschnitt im Report berechnen die Differenz A − B nur über gemeinsame Szenario/Seed-Zellen, mit Konfidenzintervall. Damit lässt sich z. B. ein neuer Trainingslauf gegen den alten vergleichen.
- `check-refs` findet zusätzlich Beinahe-Duplikate (gleiches Foto neu kodiert oder beschnitten) über ArcFace-Embeddings, sofern die ONNX-Modelle vorhanden sind.
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
