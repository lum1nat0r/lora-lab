"""Command line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .audit import audit_dataset
from .comfy.client import ComfyClient, ComfyError
from .comfy.workflow import (
    WorkflowError,
    describe,
    is_ui_format,
    load_workflow,
    ui_to_api,
)
from .config import ProjectConfig, load_project
from .db import Store
from .grid import expand, summary
from .importers.fizgig import SafetensorsError, import_lora
from .runner import build_prompt, run_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lora-lab",
        description="Automated LoRA evaluation lab for ComfyUI",
    )
    parser.add_argument("--root", default=".", help="project root (contains config/)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="validate config, workflow bindings and ComfyUI")

    plan = sub.add_parser("plan", help="expand the grid and print the job count")
    plan.add_argument("--seconds-per-image", type=float, default=15.0)

    gen = sub.add_parser("generate", help="render the grid via ComfyUI")
    gen.add_argument("--limit", type=int, default=None, help="stop after N rendered images")
    gen.add_argument("--dry-run", action="store_true", help="build prompts, send nothing")
    gen.add_argument("--run-name", default=None)
    gen.add_argument("--offline", action="store_true", help="do not fetch /object_info")
    gen.add_argument("--quiet", action="store_true")

    sub.add_parser("status", help="show database progress")

    score = sub.add_parser("score", help="run scorers over stored images")
    score.add_argument("--run", default=None)
    score.add_argument("--identity", action="store_true", help="ArcFace subject fidelity")
    score.add_argument("--verdicts", default=None, help="JSON file with attribute verdicts")
    score.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_JUDGE_URL") or None,
        help="OpenAI-compatible endpoint, e.g. http://gpu:8000/v1",
    )
    score.add_argument("--vlm-model", default=os.environ.get("VLM_JUDGE_MODEL") or None)
    score.add_argument("--vlm-api-key", default=os.environ.get("VLM_API_KEY") or None)
    score.add_argument("--only-missing", action="store_true", help="skip images that already have the axis")
    score.add_argument("--max-side", type=int, default=None, help="downscale images before sending")
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--target-megapixels", type=float, default=1.0)
    score.add_argument("--quiet", action="store_true")

    report = sub.add_parser("report", help="rank LoRAs from stored scores")
    report.add_argument("--run", default=None)
    report.add_argument("--json", action="store_true")
    report.add_argument("--bootstrap", type=int, default=2000)

    audit = sub.add_parser("audit", help="audit a dataset folder")
    audit.add_argument("path")
    audit.add_argument("--against", default=None, help="check duplicates against another folder")

    sub.add_parser("check-refs", help="verify reference images are held out from training")

    discover = sub.add_parser("discover", help="scan mounted datasets and LoRAs")
    discover.add_argument("--datasets", default="/datasets")
    discover.add_argument("--loras", default="/models/loras")
    discover.add_argument("--apply", action="store_true", help="write config/local/*.yaml")
    discover.add_argument("--check", action="store_true", help="also run the held-out check")

    imp = sub.add_parser("import-fizgig", help="read training metadata from a LoRA file")
    imp.add_argument("path")
    imp.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8099)

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "serve":
        # A broken config must not turn into a Docker restart loop: the web UI
        # comes up in a degraded state instead, so the container stays alive
        # and can be fixed with `docker exec`.
        try:
            project = load_project(root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"config error: {exc} — serving a degraded dashboard", file=sys.stderr)
            from .web.broken import serve_broken

            serve_broken(root, exc, args.host, args.port)
            return 0
    else:
        try:
            project = load_project(root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2

    if args.command == "doctor":
        return _doctor(project)
    if args.command == "plan":
        return _plan(project, args.seconds_per_image)
    if args.command == "generate":
        return _generate(project, args)
    if args.command == "status":
        return _status(project)
    if args.command == "score":
        return _score(project, args)
    if args.command == "report":
        return _report(project, args)
    if args.command == "audit":
        return _audit(Path(args.path), Path(args.against) if args.against else None)
    if args.command == "check-refs":
        return _check_refs(project)
    if args.command == "discover":
        return _discover(project, args)
    if args.command == "import-fizgig":
        return _import_fizgig(Path(args.path), args.json)
    if args.command == "serve":
        from .web.app import serve

        serve(project, host=args.host, port=args.port)
        return 0
    return 1


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _grid_hash(project: ProjectConfig) -> str:
    path = project.root / "config" / "grid.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_api(project: ProjectConfig, object_info: dict | None):
    workflow_path = project.path(project.grid.workflow_for(project.grid.profile))
    raw = load_workflow(workflow_path)
    return raw, (ui_to_api(raw, object_info) if is_ui_format(raw) else raw)


def _doctor(project: ProjectConfig) -> int:
    print(f"root        {project.root}")
    print(f"comfyui     {project.comfyui_url}")
    print(f"database    {project.db_path}")
    print(f"grid        {project.grid.name}  profile={project.grid.profile}")
    print()

    object_info = None
    try:
        with ComfyClient(project.comfyui_url, timeout=5.0) as client:
            client.ping()
            object_info = client.object_info()
        print("comfyui     ok")
    except ComfyError as exc:
        print(f"comfyui     unreachable ({exc}) — continuing with fallback specs")

    try:
        raw, api = _load_api(project, object_info)
    except (WorkflowError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"workflow    FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"workflow    {project.grid.workflow_for(project.grid.profile)}")
    print(f"nodes       {len(api)}")
    for line in describe(api):
        print("  " + line)

    scenario_list = project.scenario_list()
    jobs = expand(project.grid, scenario_list, project.subjects)
    print(f"grid        {len(jobs)} jobs · {len(scenario_list)} scenarios")

    sample = jobs[0] if jobs else None
    if sample is not None:
        try:
            prompt, job_hash = build_prompt(sample, project, api, object_info=object_info)
        except WorkflowError as exc:
            print(f"bindings    FAILED: {exc}", file=sys.stderr)
            return 2
        print(f"sample      {sample.label()}")
        print(f"prompt hash {job_hash[:16]}")
        for nid in sorted(prompt, key=int):
            node = prompt[nid]
            print(f"  {nid:>4} {node['class_type']:<28} {node['_meta'].get('title', '')}")

    check_paths(project)
    return 0


def check_paths(project: ProjectConfig) -> None:
    from .heldout import contains, count_images

    for subject in project.subjects.values():
        for kind, value in (("refs", subject.ref_dir), ("train", subject.train_dir)):
            if not value:
                continue
            path = project.path(value)
            if path.exists():
                print(f"{kind:<11} {subject.id:<16} ok       {count_images(path):>5} images  {path}")
            else:
                print(f"{kind:<11} {subject.id:<16} MISSING            {path}")
        if subject.ref_dir and subject.train_dir:
            if contains(project.path(subject.ref_dir), project.path(subject.train_dir)):
                print(f"WARNING     {subject.id}: reference and training folder overlap")


def _discover(project: ProjectConfig, args: argparse.Namespace) -> int:
    from .discover import (
        apply_loras,
        apply_subjects,
        discover_loras,
        discover_subjects,
        loras_yaml,
        subjects_yaml,
    )
    from .heldout import overlap

    datasets = Path(args.datasets)
    if not datasets.is_dir():
        print(f"datasets path not found: {datasets}", file=sys.stderr)
        return 2

    subjects = discover_subjects(datasets)
    if not subjects:
        print(f"no subject folders under {datasets}", file=sys.stderr)
        return 2

    print(f"# subjects under {datasets}")
    for item in subjects:
        print(
            f"#   {item['id']:<20} refs={item['ref_count']:<3} train={item['train_count']:<4}"
            f" {item['ref_dir'] or '-'}"
        )
    print(subjects_yaml(subjects))

    if args.check:
        print("# held-out check")
        for item in subjects:
            if not (item["ref_dir"] and item["train_dir"]):
                continue
            result = overlap(Path(item["ref_dir"]), Path(item["train_dir"]))
            mark = "ok" if result["clean"] else "LEAK"
            print(f"#   {item['id']:<20} {mark} duplicates={len(result['duplicates'])}")
        print()

    lora_root = Path(args.loras)
    loras: list[dict] = []
    if lora_root.is_dir():
        loras = discover_loras(lora_root, subjects=[item["id"] for item in subjects])
        print(f"# {len(loras)} LoRA files under {lora_root}")
        for item in loras:
            print(
                f"#   {item['id']:<32} ep{item['epoch']:<4} rank={item['rank'] or '?'}"
                f" {item['file']}"
            )
        print(loras_yaml(loras))
    else:
        print(f"# loras path not found: {lora_root}", file=sys.stderr)

    if args.apply:
        local = project.root / "config" / "local"
        apply_subjects(local / "subjects.yaml", subjects)
        print(f"wrote {local / 'subjects.yaml'}")
        if loras:
            apply_loras(local / "loras.yaml", loras)
            print(f"wrote {local / 'loras.yaml'}")

        grid_path = local / "grid.yaml"
        from .discover import apply_grid, grid_source

        defaults = os.environ.get("LORA_LAB_DEFAULTS")
        source = grid_source(project.root, Path(defaults) if defaults else None)
        if source is not None:
            apply_grid(grid_path, source, {item["id"] for item in subjects})
            print(f"wrote {grid_path} (from {source})")
        if grid_path.exists():
            print("# grid is ready: loras_file points at config/local/loras.yaml")
    return 0


def _check_refs(project: ProjectConfig) -> int:
    from .heldout import overlap

    checked = 0
    leaks = 0
    skipped = 0
    for subject in project.subjects.values():
        if not (subject.ref_dir and subject.train_dir):
            skipped += 1
            continue
        ref = project.path(subject.ref_dir)
        train = project.path(subject.train_dir)
        if not ref.is_dir() or not train.is_dir():
            print(f"{subject.id:<16} SKIP   missing folder ({ref} / {train})")
            skipped += 1
            continue

        result = overlap(ref, train)
        checked += 1
        status = "ok" if result["clean"] else "LEAK"
        print(
            f"{subject.id:<16} {status:<5} refs={result['ref_images']:<3}"
            f" train={result['train_images']:<4} duplicates={len(result['duplicates'])}"
        )
        if not result["clean"]:
            leaks += 1
            for item in result["duplicates"][:10]:
                print(f"    {item['ref']}  ==  {item['train']}")
            if len(result["duplicates"]) > 10:
                print(f"    ... {len(result['duplicates']) - 10} more")

    print(f"\n{checked} subjects checked · {leaks} with leaks · {skipped} skipped")
    return 0 if leaks == 0 else 1


def _plan(project: ProjectConfig, seconds_per_image: float) -> int:
    scenario_list = project.scenario_list()
    jobs = expand(project.grid, scenario_list, project.subjects)
    counts = summary(jobs)
    print(f"grid {project.grid.name} · profile {project.grid.profile}")
    print(f"scenarios  {len(scenario_list)}: {', '.join(s.id for s in scenario_list)}")
    print(f"loras      {len(project.grid.loras)} entries")
    print(f"seeds      {len(project.grid.seeds)}: {project.grid.seeds}")
    for condition, count in counts.items():
        if condition == "total":
            continue
        print(f"  {condition:<10} {count}")
    print(f"total      {counts['total']} images")
    minutes = counts["total"] * seconds_per_image / 60
    print(f"estimate   {minutes:.0f} min at {seconds_per_image:g}s/image")
    return 0


def _generate(project: ProjectConfig, args: argparse.Namespace) -> int:
    scenario_list = project.scenario_list()
    jobs = expand(project.grid, scenario_list, project.subjects)
    run_name = args.run_name or project.grid.name
    store = Store(project.db_path)
    run_id = store.upsert_run(
        run_name,
        _grid_hash(project),
        project.grid.profile,
        {"scenarios": [s.id for s in scenario_list], "jobs": len(jobs)},
    )

    client = ComfyClient(project.comfyui_url)
    object_info = None
    if not args.offline and not args.dry_run:
        try:
            client.ping()
            object_info = client.object_info()
        except ComfyError as exc:
            print(f"comfyui error: {exc}", file=sys.stderr)
            return 2

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        stats = run_jobs(
            project,
            store,
            jobs,
            client,
            run_id=run_id,
            limit=args.limit,
            dry_run=args.dry_run,
            on_progress=progress,
            object_info=object_info,
        )
    except KeyboardInterrupt:
        print("\ninterrupted — rerun to continue (finished jobs are skipped)", file=sys.stderr)
        return 130
    finally:
        client.close()
        store.close()

    print(
        f"done {stats.done} · skipped {stats.skipped} · failed {stats.failed}"
        f" · seen {stats.total}",
        flush=True,
    )
    return 0 if stats.failed == 0 else 1


def _status(project: ProjectConfig) -> int:
    if not project.db_path.exists():
        print("no database yet — run `lora-lab generate` first")
        return 0
    with Store(project.db_path) as store:
        counts = store.counts()
        print(f"database   {project.db_path}")
        print(f"jobs       {counts.get('total', 0)}")
        for status in ("done", "running", "pending", "failed"):
            print(f"  {status:<8} {counts.get(status, 0)}")
        print(f"images     {store.image_count()}")
        for run in store.runs():
            print(f"run        {run['name']}  profile={run['profile']}")
    return 0


def _score(project: ProjectConfig, args: argparse.Namespace) -> int:
    from .scorers.adherence import AdherenceScorer, JsonVerdicts
    from .scorers.defects import HealthScorer
    from .scorers.run import score_run

    if not project.db_path.exists():
        print("no database yet — run `lora-lab generate` first", file=sys.stderr)
        return 2

    scorers = [HealthScorer(target_megapixels=args.target_megapixels)]
    attributes = {s.id: s.attributes for s in project.scenarios.values()}

    if args.verdicts and args.vlm_url:
        print("use either --verdicts or --vlm-url, not both", file=sys.stderr)
        return 2

    vlm_judge = None
    if args.verdicts:
        judge = JsonVerdicts(Path(args.verdicts))
        scorers.append(AdherenceScorer(judge, attributes))
    elif args.vlm_url:
        if not args.vlm_model:
            print("--vlm-url needs --vlm-model", file=sys.stderr)
            return 2
        from .scorers.vlm import VLMJudge, VLMSettings

        vlm_judge = VLMJudge(
            VLMSettings(
                base_url=args.vlm_url,
                model=args.vlm_model,
                api_key=args.vlm_api_key,
                max_side=args.max_side,
            )
        )
        scorers.append(AdherenceScorer(vlm_judge, attributes))

    if args.identity:
        from .scorers.identity import (
            ArcFaceEmbedder,
            IdentityScorer,
            build_references,
            calibrate_from_references,
        )

        directories = {
            subject.id: project.path(subject.ref_dir)
            for subject in project.subjects.values()
            if subject.ref_dir and project.path(subject.ref_dir).is_dir()
        }
        if len(directories) < 2:
            print(
                "identity scoring needs reference folders for at least two subjects"
                " (one to match, one to calibrate against)",
                file=sys.stderr,
            )
            return 2
        try:
            embedder = ArcFaceEmbedder()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        references = build_references(directories, embedder)
        try:
            calibration = calibrate_from_references(references)
        except ValueError as exc:
            print(f"calibration failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"calibration  slope={calibration.slope:.2f} intercept={calibration.intercept:.2f}"
            f" genuine={calibration.genuine} impostor={calibration.impostor}"
        )
        scorers.append(IdentityScorer(references, calibration, embedder))

    with Store(project.db_path) as store:
        run_id = None
        if args.run:
            run = store.run_by_name(args.run)
            if run is None:
                print(f"unknown run: {args.run}", file=sys.stderr)
                return 2
            run_id = int(run["id"])

        def progress(message: str) -> None:
            if not args.quiet:
                print(message, flush=True)

        counts = score_run(
            project,
            store,
            scorers,
            run_id=run_id,
            limit=args.limit,
            skip_existing=args.only_missing,
            on_progress=progress,
        )

    if vlm_judge is not None:
        stats = vlm_judge.stats
        print(
            f"vlm        {stats['parsed']}/{stats['requests']} parsed · "
            f"{stats['refused']} refused · {stats['errors']} errors"
        )
        vlm_judge.close()

    print(" · ".join(f"{axis} {value}" for axis, value in sorted(counts.items())))
    return 0


def _report(project: ProjectConfig, args: argparse.Namespace) -> int:
    from .analyze.report import build_report, format_results, report_json, samples_from_store
    from .analyze.stats import analyse_all

    if not project.db_path.exists():
        print("no database yet — run `lora-lab generate` first", file=sys.stderr)
        return 2

    with Store(project.db_path) as store:
        run_id = None
        if args.run:
            run = store.run_by_name(args.run)
            if run is None:
                print(f"unknown run: {args.run}", file=sys.stderr)
                return 2
            run_id = int(run["id"])

        if args.json:
            print(report_json(build_report(store, run_id=run_id, bootstrap=args.bootstrap)))
            return 0

        samples = samples_from_store(store, run_id)
        results = analyse_all(samples, bootstrap=args.bootstrap)

    if not results:
        print("no scored images yet — run `lora-lab score` first", file=sys.stderr)
        return 2

    print(format_results(results))
    print()
    print(
        f"{len(samples)} scored images · ranking by LCB95 · "
        "mean/rob/stab are scenario-weighted"
    )
    for result in results:
        worst = sorted(result.per_scenario.items(), key=lambda item: item[1])[:2]
        if worst:
            print(
                f"  {result.lora_id[:28]:<28} weakest: "
                + ", ".join(f"{name} {value:.2f}" for name, value in worst)
            )
    return 0


def _audit(path: Path, against: Path | None = None) -> int:
    try:
        result = audit_dataset(path)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 2
    if against is not None:
        from .heldout import overlap

        result["overlap"] = overlap(path, against)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _import_fizgig(path: Path, as_json: bool) -> int:
    from .importers.fizgig import normalise_run, read_runs

    if path.is_dir() or path.name == "runs.json":
        try:
            runs = [normalise_run(run) for run in read_runs(path)]
        except (FileNotFoundError, SafetensorsError, json.JSONDecodeError) as exc:
            print(f"import error: {exc}", file=sys.stderr)
            return 2
        if as_json:
            print(json.dumps(runs, indent=2, ensure_ascii=False))
            return 0
        print(f"{len(runs)} runs")
        print(f"{'started':<17} {'family':<12} {'lora':<28} {'result':<9} settings")
        for run in runs[-40:]:
            import datetime

            started = run["started_at"]
            try:
                stamp = datetime.datetime.fromtimestamp(float(started)).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError):
                stamp = "?"
            settings = run["settings"]
            rank = settings.get("network_dim") or settings.get("lora_rank") or "?"
            lr = settings.get("learning_rate") or "?"
            epochs = settings.get("max_train_epochs") or settings.get("epochs") or "?"
            print(
                f"{stamp:<17} {str(run['family'])[:12]:<12} {str(run['lora'])[:28]:<28}"
                f" {str(run['result']):<9} rank={rank} lr={lr} epochs={epochs}"
            )
        return 0

    try:
        record = import_lora(path)
    except (FileNotFoundError, SafetensorsError) as exc:
        print(f"import error: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    print(f"file        {record['file']}")
    for key, value in record.items():
        if key in {"metadata", "file"}:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key:<20} {value}")
    if not any(k.startswith("ss_") for k in record["metadata"]):
        print("\nnote: no ss_* training metadata found in this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
