"""
FairMOP Pipeline Orchestrator.

Coordinates the execution of the four FairMOP modules in sequence:

    1. **Input Specifications** – Validate config, resolve keys, print summary
    2. **Generation** – Execute the hyperparameter grid, generate images
    3. **Evaluation** – Annotate with VLM, compute fairness & utility metrics
    4. **Output** – Build Pareto frontier, export results, optionally launch dashboard

This is the main entry point for running a complete FairMOP experiment.
"""

from __future__ import annotations

import glob
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from fairmop.config import ExperimentConfig
from fairmop.evaluation.fairness import compute_fairness_metrics
from fairmop.evaluation.vlm_judge import VLMJudge

load_dotenv()  # Load .env file (OPENAI_API_KEY, etc.)
from fairmop.generation.registry import GeneratorRegistry  # noqa: E402
from fairmop.input_specs import (  # noqa: E402
    print_experiment_summary,
    validate_or_raise,
)
from fairmop.output.export import export_results_csv, export_results_json  # noqa: E402
from fairmop.output.pareto import build_pareto_from_results  # noqa: E402


class FairMOPPipeline:
    """Main pipeline orchestrator for FairMOP experiments.

    Parameters:
        config: Experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.generator = None
        self.vlm_judge = None
        self.results: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

    # ── Module 1: Input Specifications ───────────────────────────────────

    def _setup_input_specs(self) -> None:
        """Validate configuration and print experiment summary."""
        print("\n" + "=" * 60)
        print("  MODULE 1 – Input Specifications")
        print("=" * 60)

        validate_or_raise(self.config)
        print_experiment_summary(self.config)

    # ── Module 2: Generation ─────────────────────────────────────────────

    def _setup_generator(self) -> None:
        """Initialize the T2I model generator backend."""
        print("\n" + "=" * 60)
        print("  MODULE 2 – Generation")
        print("=" * 60)

        device = "cpu"
        if self.config.gpu_index is not None:
            from fairmop.utils import set_gpu_device

            device = set_gpu_device(self.config.gpu_index)

        self.generator = GeneratorRegistry.create(
            name=self.config.model_name,
            device=device,
            **self.config.model_params,
        )
        self.generator.setup()
        print(f"[Pipeline] Generator ready: {self.generator}")

    def _run_generation(self) -> str:
        """Execute the hyperparameter grid and generate all images.

        Returns:
            Path to the directory containing generated images.
        """
        output_dir = os.path.join(self.config.output_dir, "images")
        os.makedirs(output_dir, exist_ok=True)

        configurations = self.config.hyperparameter_grid.configurations()
        total_configs = len(configurations)
        total_images = self.config.total_images()

        print(
            f"[Pipeline] Generating {total_images} images "
            f"across {total_configs} configurations..."
        )

        prompt_slug = self.generator.slugify_prompt(self.config.prompt)
        generated_count = 0

        for cfg_idx, hyperparams in enumerate(configurations):
            params_slug = self.generator.slugify_params(hyperparams)

            print(f"\n[Config {cfg_idx + 1}/{total_configs}] {params_slug}")

            for img_idx in range(self.config.num_images_per_config):
                seed = self.config.seed_start + img_idx

                # Check if image already exists (resume support)
                filename = f"{prompt_slug}_{params_slug}_seed{seed}.png"
                filepath = os.path.join(output_dir, filename)

                if os.path.exists(filepath):
                    generated_count += 1
                    print(
                        f"  [{generated_count}/{total_images}] "
                        f"Seed {seed} – already exists, skipping"
                    )
                    continue

                try:
                    t0 = time.time()
                    image = self.generator.generate(
                        prompt=self.config.prompt,
                        seed=seed,
                        **hyperparams,
                    )
                    self.generator.save_image(
                        image, output_dir, prompt_slug, params_slug, seed
                    )
                    generated_count += 1
                    elapsed = time.time() - t0

                    remaining = total_images - generated_count
                    eta_secs = remaining * elapsed
                    eta_min = eta_secs / 60

                    print(
                        f"  [{generated_count}/{total_images}] "
                        f"Seed {seed} ✓  ({elapsed:.1f}s) "
                        f"– ETA: {eta_min:.0f}min"
                    )

                except Exception as e:
                    generated_count += 1  # count to keep progress accurate
                    print(
                        f"  [{generated_count}/{total_images}] Seed {seed} ERROR: {e}"
                    )
                    continue

        self.generator.teardown()
        print(
            f"\n[Pipeline] Generation complete: {generated_count} images "
            f"saved to {output_dir}"
        )

        return output_dir

    # ── Module 3: Evaluation ─────────────────────────────────────────────

    def _run_evaluation(self, images_dir: str) -> Dict[str, Any]:
        """Evaluate all generated images for fairness and utility metrics.

        Parameters:
            images_dir: Directory containing generated images.

        Returns:
            Results dictionary with per-configuration metrics.
        """
        print("\n" + "=" * 60)
        print("  MODULE 3 – Evaluation")
        print("=" * 60)

        # Discover and group images by configuration
        config_groups = self._discover_images(images_dir)
        if not config_groups:
            raise ValueError(f"No valid images found in {images_dir}")

        topic = None
        results = {"configurations": {}, "summary": {}}

        # Setup VLM judge if needed
        needs_vlm = any(
            m in {"entropy", "kl", "entropy_fairness", "kl_fairness"}
            for m in self.config.metrics
        )
        if needs_vlm:
            self.vlm_judge = VLMJudge(
                provider=self.config.vlm_provider,
                model_name=self.config.vlm_model,
                api_key=self.config.api_key,
            )
            self.vlm_judge.setup()

        # Setup CLIP model if needed
        clip_model = None
        clip_preprocess = None
        device = "cpu"
        if self.config.gpu_index is not None:
            from fairmop.utils import set_gpu_device

            device = set_gpu_device(self.config.gpu_index)

        needs_clip = "clip_score" in self.config.metrics
        if needs_clip:
            try:
                import open_clip
                import torch

                clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14",
                    pretrained="openai",
                    device=device,
                )
                print("[Pipeline] CLIP model loaded (ViT-L-14, open_clip)")
            except ImportError:
                print(
                    "[Pipeline] WARNING: open_clip not available, skipping clip_score"
                )  # noqa: E501
                needs_clip = False

        total_configs = len(config_groups)
        processed = 0

        for config_key, image_paths in config_groups.items():
            processed += 1
            print(
                f"\n[Eval {processed}/{total_configs}] {config_key} "
                f"({len(image_paths)} images)"
            )

            config_result = {
                "images": len(image_paths),
                "demographics": [],
                "aggregates": {},
            }

            # Extract topic from first image
            if topic is None:
                name, _, _ = self._parse_filename(os.path.basename(image_paths[0]))
                topic = name or "unknown"

            # ── VLM Annotation ───────────────────────────────────────
            if needs_vlm and self.vlm_judge:
                print(f"  Annotating {len(image_paths)} images with VLM...")
                annotations = self.vlm_judge.annotate_batch(image_paths)
                config_result["demographics"] = (
                    self.vlm_judge.annotations_to_legacy_format(annotations)
                )

                # Compute fairness metrics
                fairness = compute_fairness_metrics(
                    annotations,
                    protected_attribute=self.config.protected_attribute,
                )

                config_result["aggregates"].update(
                    {
                        "gender_entropy": fairness["gender_entropy"],
                        "ethnicity_entropy": fairness["ethnicity_entropy"],
                        "age_entropy": fairness["age_entropy"],
                        "gender_kl": fairness["gender_kl"],
                        "ethnicity_kl": fairness["ethnicity_kl"],
                        "age_kl": fairness["age_kl"],
                        "primary_entropy": fairness["primary_entropy"],
                        "primary_kl": fairness["primary_kl"],
                        "entropy_fairness": (
                            fairness["gender_entropy"]
                            + fairness["ethnicity_entropy"]
                            + fairness["age_entropy"]
                        )
                        / 3.0,
                        "gender_distribution": fairness["gender_distribution"],
                        "ethnicity_distribution": fairness["ethnicity_distribution"],
                        "age_distribution": fairness["age_distribution"],
                    }
                )

                print(
                    f"  Fairness ({self.config.protected_attribute} entropy): "
                    f"{fairness['primary_entropy']:.4f}"
                )

            # ── CLIP Score ───────────────────────────────────────────
            if needs_clip and clip_model is not None:
                import open_clip
                import torch
                from PIL import Image

                scores = []
                tokenizer = open_clip.get_tokenizer("ViT-L-14")
                text_tokens = tokenizer([self.config.prompt]).to(device)

                for path in image_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        tensor = clip_preprocess(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            img_feat = clip_model.encode_image(tensor)
                            txt_feat = clip_model.encode_text(text_tokens)
                            img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
                            txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True)
                            sim = torch.cosine_similarity(img_feat, txt_feat, dim=1)
                            scores.append(sim.item())
                    except Exception as e:
                        print(f"    CLIP error for {os.path.basename(path)}: {e}")

                if scores:
                    import numpy as np

                    config_result["aggregates"]["avg_clip_score"] = float(
                        np.mean(scores)
                    )
                    config_result["aggregates"]["std_clip_score"] = float(
                        np.std(scores)
                    )
                    config_result["clip_scores"] = scores
                    print(f"  CLIP Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            results["configurations"][config_key] = config_result

        results["topic"] = topic
        results["summary"] = {
            "total_configs": len(config_groups),
            "total_images": sum(len(p) for p in config_groups.values()),
            "metrics": self.config.metrics,
            "prompt": self.config.prompt,
            "model": self.config.model_name,
            "protected_attribute": self.config.protected_attribute,
        }

        return results

    # ── Module 4: Output ─────────────────────────────────────────────────

    def _run_output(self, results: Dict[str, Any]) -> None:
        """Export results and perform Pareto analysis.

        Parameters:
            results: Evaluation results dictionary.
        """
        print("\n" + "=" * 60)
        print("  MODULE 4 – Output")
        print("=" * 60)

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Export JSON
        json_path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_results.json",
        )
        export_results_json(results, json_path)

        # Export CSV
        csv_path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_summary.csv",
        )
        export_results_csv(results, csv_path)

        # Pareto analysis
        aggregates_sample = (
            list(results["configurations"].values())[0]["aggregates"]
            if results["configurations"]
            else {}
        )

        utility_key = None
        fairness_key = None

        for key in ["avg_clip_score", "inverse_fid", "precision"]:
            if key in aggregates_sample:
                utility_key = key
                break

        for key in ["primary_entropy", "gender_entropy", "entropy_fairness"]:
            if key in aggregates_sample:
                fairness_key = key
                break

        if utility_key and fairness_key:
            pareto_result = build_pareto_from_results(
                results,
                utility_metric=utility_key,
                fairness_metric=fairness_key,
            )
            print(f"\n{pareto_result.summary()}")

            from fairmop.output.export import export_pareto_csv

            pareto_csv = os.path.join(
                self.config.output_dir,
                f"{self.config.experiment_name}_pareto.csv",
            )
            export_pareto_csv(pareto_result, pareto_csv)
        else:
            print("[Pipeline] Insufficient metrics for Pareto analysis.")

        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"\n{'=' * 60}")
        print("  FairMOP Pipeline Complete!")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"{'=' * 60}\n")

    # ── Main Run ─────────────────────────────────────────────────────────

    def run(
        self,
        skip_generation: bool = False,
        skip_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full FairMOP pipeline.

        Parameters:
            skip_generation: Skip Module 2 (use existing images).
            skip_evaluation: Skip Module 3 (only generate images).

        Returns:
            The final results dictionary.
        """
        self._start_time = time.time()

        # Module 1
        self._setup_input_specs()

        # Module 2
        if not skip_generation:
            self._setup_generator()
            images_dir = self._run_generation()
        else:
            images_dir = os.path.join(self.config.output_dir, "images")
            if not os.path.isdir(images_dir):
                raise FileNotFoundError(
                    f"Images directory not found: {images_dir}. "
                    f"Run generation first or provide correct path."
                )
            print(f"\n[Pipeline] Skipping generation. Using images from: {images_dir}")

        # Module 3
        if not skip_evaluation:
            self.results = self._run_evaluation(images_dir)
        else:
            print("\n[Pipeline] Skipping evaluation.")
            self.results = {"topic": "unknown", "configurations": {}, "summary": {}}

        # Module 4
        if not skip_evaluation:
            self._run_output(self.results)

        return self.results

    # ── Standalone Evaluation ────────────────────────────────────────────

    @staticmethod
    def evaluate_only(
        images_dir: str,
        metrics: List[str],
        prompt: str = "the face of a nurse",
        vlm_provider: str = "openai",
        vlm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        output_path: str = "fairmop_results.json",
        gpu_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on pre-existing images without generation.

        Convenience method for evaluating images that were generated
        outside the FairMOP pipeline.

        Parameters:
            images_dir: Path to the images directory.
            metrics: List of metrics to compute.
            prompt: Text prompt used for generation.
            vlm_provider: VLM provider for annotations.
            vlm_model: VLM model name.
            api_key: API key for the VLM.
            output_path: Path for the output JSON file.
            gpu_index: GPU device index.

        Returns:
            Results dictionary.
        """
        config = ExperimentConfig(
            prompt=prompt,
            metrics=metrics,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model
            or (
                "gpt-4o-2024-05-13" if vlm_provider == "openai" else "gemini-2.0-flash"
            ),
            api_key=api_key,
            gpu_index=gpu_index,
            output_dir=os.path.dirname(os.path.abspath(output_path)),
            experiment_name=os.path.splitext(os.path.basename(output_path))[0],
        )

        pipeline = FairMOPPipeline(config)
        pipeline._start_time = time.time()
        pipeline._setup_input_specs()

        results = pipeline._run_evaluation(images_dir)
        pipeline.results = results
        pipeline._run_output(results)

        return results

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _discover_images(images_dir: str) -> Dict[str, List[str]]:
        """Discover and group images by configuration.

        Parameters:
            images_dir: Root directory containing images.

        Returns:
            Dict mapping configuration keys to lists of image paths.
        """
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for ext in extensions:
            pattern = os.path.join(images_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))

        config_groups: Dict[str, List[str]] = {}

        for path in sorted(image_files):
            filename = os.path.basename(path)
            name, params, seed = FairMOPPipeline._parse_filename(filename)

            if not name or not params:
                continue

            config_key = f"{name}_{params}"
            config_groups.setdefault(config_key, []).append(path)

        return config_groups

    @staticmethod
    def _parse_filename(filename: str):
        """Parse a FairMOP-formatted filename.

        Supports formats:
            - ``topic_params_seedX.ext``
            - ``topic_cfg0.0_editguidance0.0_fairdifTrue_seedX.ext``

        Returns:
            Tuple of (topic, params, seed) or (None, None, None).
        """
        base = os.path.splitext(filename)[0]
        parts = base.split("_")

        if len(parts) < 2:
            return None, None, None

        name = parts[0]
        seed_match = re.search(r"seed(\d+)", base)
        seed = seed_match.group(1) if seed_match else None

        if seed_match:
            params = base[len(name) + 1 : seed_match.start()]
            params = params.rstrip("_")
        else:
            params = "_".join(parts[1:])

        return name, params, seed
