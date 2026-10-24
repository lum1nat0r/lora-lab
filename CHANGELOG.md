# Changelog

Nennenswerte Änderungen an lora-lab. Format nach
[Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach
[SemVer](https://semver.org/lang/de/). Ältere Änderungen stehen in der
Git-Historie.

## [Unreleased]

### Fixed

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
