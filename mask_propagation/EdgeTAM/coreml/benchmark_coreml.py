#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark EdgeTAM PyTorch vs CoreML performance.

Author: Krish Mehta (https://github.com/DjKesu)
"""

import os
import sys
import time
from statistics import mean, stdev

import coremltools as ct
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_pytorch_model():
    """Load PyTorch EdgeTAM model."""
    device = torch.device("cpu")
    GlobalHydra.instance().clear()

    config_path = os.path.abspath("sam2/configs/edgetam.yaml")
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, "checkpoints/edgetam.pt")
        model = model.to(device)
        model.eval()

    return model


def load_coreml_models():
    """Load CoreML models."""
    image_encoder = ct.models.MLModel("./coreml_models/edgetam_image_encoder.mlpackage")
    prompt_encoder = ct.models.MLModel(
        "./coreml_models/edgetam_prompt_encoder.mlpackage"
    )
    mask_decoder = ct.models.MLModel("./coreml_models/edgetam_mask_decoder.mlpackage")

    return image_encoder, prompt_encoder, mask_decoder


def create_test_image():
    """Create a test image with shapes."""
    image = Image.new("RGB", (1024, 1024), color="lightblue")
    draw = ImageDraw.Draw(image)
    draw.ellipse([200, 200, 400, 400], fill="red", outline="darkred", width=3)
    draw.rectangle([600, 300, 800, 500], fill="green", outline="darkgreen", width=3)
    return image


def benchmark_pytorch(model, image, point_x, point_y, num_runs=10):
    """Benchmark PyTorch model."""
    predictor = SAM2ImagePredictor(model)

    # Warmup run
    predictor.set_image(np.array(image))
    _ = predictor.predict(
        point_coords=np.array([[point_x, point_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # Pre-encode image once (fair comparison with CoreML)
    predictor.set_image(np.array(image))

    times = []
    for i in range(num_runs):
        start_time = time.perf_counter()

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[point_x, point_y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    return times, best_mask, best_score


def benchmark_coreml(models, image, point_x, point_y, pytorch_model, num_runs=10):
    """Benchmark CoreML models - each model handles its own processing."""
    image_encoder, prompt_encoder, mask_decoder = models

    # Pre-compute image encoding and positional encoding (done once per image in real usage)
    encoder_out = image_encoder.predict({"image": image})

    # Get proper positional encoding from pre-loaded PyTorch model
    with torch.no_grad():
        dense_pe = pytorch_model.sam_prompt_encoder.get_dense_pe()
    image_pe = dense_pe.detach().numpy()

    # Warmup run
    point_coords = np.zeros((1, 4, 2), dtype=np.float32)
    point_labels = np.full((1, 4), -1, dtype=np.float32)
    point_coords[0, 0] = [point_x, point_y]
    point_labels[0, 0] = 1

    _ = prompt_encoder.predict(
        {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "boxes": np.zeros((1, 4), dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        }
    )

    times = []
    for i in range(num_runs):
        start_time = time.perf_counter()

        # Time only the prompt encoding + mask decoding (equivalent to PyTorch predict)
        point_coords = np.zeros((1, 4, 2), dtype=np.float32)
        point_labels = np.full((1, 4), -1, dtype=np.float32)
        point_coords[0, 0] = [point_x, point_y]
        point_labels[0, 0] = 1

        prompt_out = prompt_encoder.predict(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "boxes": np.zeros((1, 4), dtype=np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            }
        )

        mask_out = mask_decoder.predict(
            {
                "image_embeddings": encoder_out["vision_features"],
                "image_pe": image_pe,
                "sparse_prompt_embeddings": prompt_out["sparse_embeddings"],
                "dense_prompt_embeddings": prompt_out["dense_embeddings"],
                "high_res_feat_0": encoder_out["high_res_feat_0"],
                "high_res_feat_1": encoder_out["high_res_feat_1"],
                "multimask_output": np.array([1.0], dtype=np.float32),
            }
        )

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    # Get best mask from final run
    masks = mask_out["masks"]
    iou_scores = mask_out["iou_pred"]
    best_idx = np.argmax(iou_scores)
    best_mask = masks[0, best_idx]
    best_score = iou_scores[0, best_idx]

    # Scale mask to original size
    from scipy import ndimage

    best_mask = ndimage.zoom(best_mask, (1024 / 256, 1024 / 256), order=1)
    best_mask = (best_mask > 0.0).astype(np.float32)

    return times, best_mask, best_score


def run_benchmark():
    """Run comprehensive benchmark."""
    print("EdgeTAM Performance Benchmark")
    print("=" * 50)

    # Create test image and point
    test_image = create_test_image()
    point_x, point_y = 300, 300
    num_runs = 5

    print(f"Test setup: {num_runs} runs, point at ({point_x}, {point_y})")
    print()

    # Run PyTorch benchmark
    print("=== PYTORCH BENCHMARK ===")
    pytorch_model = load_pytorch_model()
    pt_times, pt_mask, pt_score = benchmark_pytorch(
        pytorch_model, test_image, point_x, point_y, num_runs
    )
    pt_mean = mean(pt_times)
    pt_std = stdev(pt_times) if len(pt_times) > 1 else 0
    pt_coverage = pt_mask.sum() / (1024 * 1024) * 100

    print(f"PyTorch Results:")
    print(f"  Average time: {pt_mean:.1f} ± {pt_std:.1f} ms")
    print(f"  IoU Score: {pt_score:.4f}")
    print(f"  Coverage: {pt_coverage:.1f}%")
    print()

    # Run CoreML benchmark
    print("=== COREML BENCHMARK ===")
    coreml_models = load_coreml_models()
    cm_times, cm_mask, cm_score = benchmark_coreml(
        coreml_models, test_image, point_x, point_y, pytorch_model, num_runs
    )
    cm_mean = mean(cm_times)
    cm_std = stdev(cm_times) if len(cm_times) > 1 else 0
    cm_coverage = cm_mask.sum() / (1024 * 1024) * 100

    print(f"CoreML Results:")
    print(f"  Average time: {cm_mean:.1f} ± {cm_std:.1f} ms")
    print(f"  IoU Score: {cm_score:.4f}")
    print(f"  Coverage: {cm_coverage:.1f}%")
    print()

    # Performance comparison
    speedup = pt_mean / cm_mean if cm_mean > 0 else 0
    score_diff = abs(pt_score - cm_score)
    coverage_diff = abs(pt_coverage - cm_coverage)

    print("Performance Comparison:")
    print("=" * 30)
    print(f"Speed comparison:")
    if speedup > 1:
        print(f"  CoreML is {speedup:.1f}x FASTER than PyTorch")
    else:
        print(f"  PyTorch is {1/speedup:.1f}x FASTER than CoreML")

    print(f"Quality comparison:")
    print(f"  Score difference: {score_diff:.4f}")
    print(f"  Coverage difference: {coverage_diff:.1f}%")

    print(f"Model size comparison:")
    print(f"  PyTorch: 54MB")
    print(f"  CoreML: 21.4MB (60% smaller)")

    # Overall assessment
    print()
    print("Overall Assessment:")
    print("=" * 20)
    if score_diff < 0.01 and coverage_diff < 1.0:
        print("Quality: Excellent parity between PyTorch and CoreML")
    elif score_diff < 0.05 and coverage_diff < 2.0:
        print("Quality: Good parity between PyTorch and CoreML")
    else:
        print("Quality: Some differences detected")

    if speedup > 1.2:
        print("Performance: CoreML shows significant speed improvement")
    elif speedup > 0.8:
        print("Performance: Comparable speed between implementations")
    else:
        print("Performance: PyTorch faster than CoreML")

    print("Size: CoreML models are 60% smaller (21.4MB vs 54MB)")
    print()

    return {
        "pytorch_time_ms": pt_mean,
        "coreml_time_ms": cm_mean,
        "speedup": speedup,
        "pytorch_score": pt_score,
        "coreml_score": cm_score,
        "score_diff": score_diff,
    }


if __name__ == "__main__":
    results = run_benchmark()
