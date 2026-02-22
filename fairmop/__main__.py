"""
Entry point for running FairMOP as a module: ``python -m fairmop``

Supports four modes:
    1. Run pipeline from a YAML config:
        python -m fairmop run --config experiment.yaml

    2. Launch the Streamlit dashboard:
        python -m fairmop dashboard

    3. Evaluate pre-existing images:
        python -m fairmop evaluate --images ./images --metrics clip_score entropy

    4. Download FairMOP pre-generated dataset from HuggingFace Hub:
        python -m fairmop download --output ./my_images/
        python -m fairmop download --output ./my_images/ --model sd
        python -m fairmop download --list-models   # see available model folders
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairmop",
        description="FairMOP – Benchmarking Fairness-Utility Trade-offs in T2I Models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run the full FairMOP pipeline from a YAML config"
    )
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the experiment YAML configuration file",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Override the output directory from the config",
    )
    run_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip image generation (use existing images)",
    )
    run_parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation (only generate images)",
    )

    # ── evaluate ────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate pre-existing images")
    eval_parser.add_argument(
        "--images",
        "-i",
        type=str,
        required=True,
        help="Path to folder containing generated images",
    )
    eval_parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["clip_score", "entropy"],
        help="Metrics to compute (clip_score, fid, prdc, entropy, kl)",
    )
    eval_parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="the face of a nurse",
        help="Text prompt used for generation (needed for CLIP score)",
    )
    eval_parser.add_argument(
        "--vlm-provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="VLM provider for demographic annotation",
    )
    eval_parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="VLM model name (e.g. gpt-4o-2024-05-13)",
    )
    eval_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set via OPENAI_API_KEY / GEMINI_API_KEY env vars)",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="fairmop_results.json",
        help="Output JSON file path",
    )
    eval_parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use",
    )

    # ── download ────────────────────────────────────────────────────────
    dl_parser = subparsers.add_parser(
        "download",
        help="Download FairMOP pre-generated image dataset from HuggingFace Hub",
    )
    dl_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./fairmop_output",
        help=(
            "Root output directory (default: ./fairmop_output). "
            "When --model is given, images are placed under "
            "<output>/<model>_evaluation/images/."
        ),
    )
    dl_parser.add_argument(
        "--repo-id",
        type=str,
        default="marconb10/FairMOP_images",
        help="HuggingFace dataset repo ID (default: marconb10/FairMOP_images)",
    )
    dl_parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Download only one model folder. Exact names available in the repo: "
            "'sd', 'sdxl', 'DeCoDi', 'Fluxdev_default', 'Fluxdev_configs'. "
            "Run with --list-models to fetch the current list. "
            "Omit to download the entire dataset."
        ),
    )
    dl_parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List all model folders available in the HuggingFace repo and exit "
            "(no download performed)."
        ),
    )
    dl_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "HuggingFace token for private repos. "
            "Alternatively set the HF_TOKEN environment variable or run "
            "`huggingface-cli login` beforehand."
        ),
    )

    # ── dashboard ───────────────────────────────────────────────────────
    dash_parser = subparsers.add_parser(
        "dashboard", help="Launch the Streamlit visualization dashboard"
    )
    dash_parser.add_argument(
        "--results",
        "-r",
        type=str,
        default=None,
        help="Pre-load a results JSON file into the dashboard",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit server",
    )

    return parser


def _list_models(repo_id: str, token: str | None) -> list[str]:
    """Return the sorted list of top-level model folders in *repo_id*."""
    import os

    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        print(
            "[FairMOP] ERROR: 'huggingface_hub' is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )
        raise SystemExit(1)

    resolved_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    )
    files = list_repo_files(repo_id, repo_type="dataset", token=resolved_token)
    return sorted(set(f.split("/")[0] for f in files if "/" in f))


def _run_download(
    repo_id: str,
    output: str,
    model: str | None,
    token: str | None,
    list_models: bool = False,
) -> None:
    """Download the FairMOP pre-generated image dataset from HuggingFace Hub.

    Layout after download
    ---------------------
    Single model (--model sd)::

        <output>/
          sd_evaluation/
            images/               ← images land here
              sd_default_nurse_seed1.jpg
              ...

    Full dataset (no --model)::

        <output>/
          DeCoDi_evaluation/images/
          Fluxdev_configs_evaluation/images/
          ...

    After the download the command prints ready-to-use ``evaluate`` commands.
    """
    import os
    import shutil

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[FairMOP] ERROR: 'huggingface_hub' is not installed.\n"
            "Install it with:  pip install huggingface_hub\n"
            "or:               pip install -e \".[datasets]\"  (recommended)"
        )
        raise SystemExit(1)

    # Prefer explicit token → env var → cached login (no interactive prompt)
    resolved_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    )

    # ── --list-models: just print available folders and exit ────────────
    if list_models:
        print(f"[FairMOP] Fetching model list from {repo_id} …")
        models = _list_models(repo_id, token)
        print(f"\nAvailable model folders ({len(models)}):")
        for m in models:
            print(f"  {m}")
        print(
            "\nUsage example:\n"
            "  python -m fairmop download --model sd"
        )
        return

    # ── validate --model against actual repo folders ────────────────────
    allow_patterns = None
    models_to_move: list[str] = []

    if model:
        available = _list_models(repo_id, token)
        if model not in available:
            print(
                f"[FairMOP] ERROR: model folder '{model}' not found in the repo.\n"
                f"Available folders: {', '.join(available)}\n"
                f"Tip: run  python -m fairmop download --list-models  to see them."
            )
            raise SystemExit(1)
        allow_patterns = [f"{model}/*"]
        models_to_move = [model]
        print(f"[FairMOP] Downloading model subset: '{model}'")
    else:
        models_to_move = _list_models(repo_id, token)

    # ── snapshot_download preserves repo folder layout ──────────────────
    # We use a staging dir so we can restructure afterwards without
    # touching the user's fairmop_output content.
    staging_dir = os.path.join(output, "_hf_staging")
    os.makedirs(staging_dir, exist_ok=True)

    print(f"[FairMOP] Repo     : {repo_id}")
    print(f"[FairMOP] Output   : {os.path.abspath(output)}")
    print()

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=staging_dir,
        allow_patterns=allow_patterns,
        token=resolved_token,
    )

    # ── move {model}/ → {output}/{model}_evaluation/images/ ────────────
    evaluation_dirs: list[str] = []

    for m in models_to_move:
        src = os.path.join(staging_dir, m)
        if not os.path.isdir(src):
            continue

        eval_dir = os.path.join(output, f"{m}_evaluation")
        images_dir = os.path.join(eval_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Move every image file from staging into images/
        moved = 0
        for fname in os.listdir(src):
            shutil.move(os.path.join(src, fname), os.path.join(images_dir, fname))
            moved += 1

        print(f"[FairMOP] {m:30s} → {images_dir}  ({moved} files)")
        evaluation_dirs.append(eval_dir)

    # Clean up staging dir
    shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n[FairMOP] Download complete.")

    # ── Print ready-to-use evaluate commands ────────────────────────────
    if evaluation_dirs:
        print("\n" + "=" * 60)
        print("  Ready-to-use evaluate commands")
        print("=" * 60)
        for eval_dir in evaluation_dirs:
            name = os.path.basename(eval_dir)          # e.g. 'sd_evaluation'
            images_path = os.path.join(eval_dir, "images")
            results_path = os.path.join(eval_dir, f"{name}_results.json")
            # Quote paths so spaces in folder names don't break the shell command
            images_q  = f'"{images_path}"'  if " " in images_path  else images_path
            results_q = f'"{results_path}"' if " " in results_path else results_path
            print(
                f"\n  # Evaluate '{name}' images\n"
                f"  python -m fairmop evaluate \\\n"
                f"      --images {images_q} \\\n"
                f"      --prompt \"the face of a nurse\" \\\n"
                f"      --metrics clip_score entropy kl \\\n"
                f"      --vlm-provider openai \\\n"
                f"      --output {results_q}"
            )
        print("\n" + "=" * 60)


def main():
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file (OPENAI_API_KEY, etc.)

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        from fairmop.config import ExperimentConfig
        from fairmop.pipeline import FairMOPPipeline

        config = ExperimentConfig.from_yaml(args.config)
        if args.output_dir:
            config.output_dir = args.output_dir

        pipeline = FairMOPPipeline(config)
        pipeline.run(
            skip_generation=args.skip_generation,
            skip_evaluation=args.skip_evaluation,
        )

    elif args.command == "evaluate":
        from fairmop.pipeline import FairMOPPipeline

        FairMOPPipeline.evaluate_only(
            images_dir=args.images,
            metrics=args.metrics,
            prompt=args.prompt,
            vlm_provider=args.vlm_provider,
            vlm_model=args.vlm_model,
            api_key=args.api_key,
            output_path=args.output,
            gpu_index=args.gpu,
        )

    elif args.command == "download":
        _run_download(
            repo_id=args.repo_id,
            output=args.output,
            model=args.model,
            token=args.token,
            list_models=args.list_models,
        )

    elif args.command == "dashboard":
        import os
        import subprocess

        dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path,
            "--server.port",
            str(args.port),
        ]
        if args.results:
            cmd.extend(["--", "--results", args.results])
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
