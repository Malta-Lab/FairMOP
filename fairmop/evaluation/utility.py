"""
Utility metrics for FairMOP.

Implements image quality and semantic alignment metrics:
    - **CLIP Score**: Cosine similarity between CLIP image and text embeddings
      (ViT-L/14). Measures semantic alignment with the prompt.
    - **FID (Fréchet Inception Distance)**: Distribution-level perceptual
      quality using InceptionV3 features and the Clean-FID library.
    - **PRDC Precision**: Fraction of generated samples falling inside
      the manifold of real images (k-NN based).

For Pareto analysis, FID is inverted (1/FID) to ensure all metrics follow
a "higher is better" convention.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


def compute_clip_score(
    image_paths: List[str],
    text_prompt: str,
    device: str = "cpu",
    clip_model: str = "ViT-L/14",
) -> Dict[str, float]:
    """Compute CLIP Score for a set of images against a text prompt.

    Parameters:
        image_paths: List of image file paths.
        text_prompt: The text prompt for comparison.
        device: PyTorch device (e.g., ``"cuda:0"``).
        clip_model: CLIP model variant.

    Returns:
        A dictionary with keys ``"mean"``, ``"std"``, ``"scores"`` (per-image).
    """
    try:
        import open_clip
        import torch
    except ImportError:
        raise ImportError(
            "open_clip_torch and torch are required for CLIP Score. "
            "Install with: pip install open_clip_torch torch"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained="openai", device=device,
    )
    tokenizer = open_clip.get_tokenizer(clip_model)
    text_tokens = tokenizer([text_prompt]).to(device)

    scores = []
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(
                    dim=1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                similarity = torch.cosine_similarity(
                    image_features, text_features, dim=1
                )
                scores.append(similarity.item())
        except Exception as e:
            warnings.warn(f"Error computing CLIP score for {path}: {e}")
            continue

    if not scores:
        return {"mean": 0.0, "std": 0.0, "scores": []}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def compute_fid(
    generated_dir: str,
    reference_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute Fréchet Inception Distance (FID).

    Uses the Clean-FID library for consistent and reproducible computation.
    All images are resized to 299×299 and processed using InceptionV3.

    Parameters:
        generated_dir: Directory with generated images.
        reference_dir: Directory with real reference images (e.g., FFHQ).
        device: PyTorch device.
        batch_size: Batch size for feature extraction.

    Returns:
        A dictionary with ``"fid"`` and ``"inverse_fid"`` (1/FID for Pareto).
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )

    try:
        fid_score = cleanfid.compute_fid(
            generated_dir,
            reference_dir,
            device=device,
            batch_size=batch_size,
        )
    except Exception as e:
        warnings.warn(f"Error computing FID: {e}")
        return {"fid": float("inf"), "inverse_fid": 0.0}

    inverse_fid = 1.0 / fid_score if fid_score > 0 else float("inf")

    return {
        "fid": float(fid_score),
        "inverse_fid": float(inverse_fid),
    }


def compute_prdc_precision(
    generated_dir: str,
    reference_dir: str,
    device: str = "cpu",
    k: int = 5,
    reference_features_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute PRDC Precision (and optionally Recall, Density, Coverage).

    Uses InceptionV3 feature space with k-NN geometry.

    Parameters:
        generated_dir: Directory with generated images.
        reference_dir: Directory with reference images.
        device: PyTorch device.
        k: Number of nearest neighbors.
        reference_features_path: Path to precomputed reference features (.npy).

    Returns:
        Dictionary with ``"precision"``, ``"recall"``, ``"density"``,
        ``"coverage"`` scores.
    """
    try:
        import torch
        from torchvision import models, transforms
    except ImportError:
        raise ImportError(
            "torch and torchvision are required for PRDC. "
            "Install with: pip install torch torchvision"
        )

    # Feature extraction setup
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False,
    )
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def extract_features(image_dir: str) -> np.ndarray:
        """Extract InceptionV3 features from a directory of images."""
        import glob

        pattern_exts = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for ext in pattern_exts:
            image_files.extend(
                glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
            )

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        features_list = []
        for path in image_files:
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = inception(tensor)
                features_list.append(feat.cpu().numpy().flatten())
            except Exception as e:
                warnings.warn(f"Error processing {path}: {e}")
                continue

        return np.array(features_list)

    # Load or compute features
    if reference_features_path and os.path.exists(reference_features_path):
        real_features = np.load(reference_features_path)
    else:
        real_features = extract_features(reference_dir)
        if reference_features_path:
            os.makedirs(os.path.dirname(reference_features_path), exist_ok=True)
            np.save(reference_features_path, real_features)

    gen_features = extract_features(generated_dir)

    # Compute PRDC metrics using k-NN
    try:
        from prdc import compute_prdc as _compute_prdc

        metrics = _compute_prdc(
            real_features=real_features,
            fake_features=gen_features,
            nearest_k=k,
        )
        return {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "density": float(metrics["density"]),
            "coverage": float(metrics["coverage"]),
        }
    except ImportError:
        # Fallback: compute precision manually
        return _compute_precision_manual(real_features, gen_features, k)


def _compute_precision_manual(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """Manual k-NN precision when the prdc package is unavailable.

    Parameters:
        real_features: Real image features array (N_real, D).
        gen_features: Generated image features array (N_gen, D).
        k: Number of nearest neighbors.

    Returns:
        Dictionary with ``"precision"`` score.
    """
    from scipy.spatial.distance import cdist

    # Compute pairwise distances
    real_real_dist = cdist(real_features, real_features, metric="euclidean")
    gen_real_dist = cdist(gen_features, real_features, metric="euclidean")

    # Real-data k-th NN radii
    np.fill_diagonal(real_real_dist, np.inf)
    real_radii = np.sort(real_real_dist, axis=1)[:, k - 1]

    # For each generated sample, check if it falls inside any real manifold ball
    min_dist_to_real = np.min(gen_real_dist, axis=1)
    nearest_real_idx = np.argmin(gen_real_dist, axis=1)
    nearest_real_radius = real_radii[nearest_real_idx]

    precision = float(np.mean(min_dist_to_real <= nearest_real_radius))

    return {
        "precision": precision,
        "recall": 0.0,
        "density": 0.0,
        "coverage": 0.0,
    }
