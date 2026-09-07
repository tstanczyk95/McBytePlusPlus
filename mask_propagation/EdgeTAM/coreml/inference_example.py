#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
EdgeTAM CoreML Video Tracking Example

Demonstrates how to use EdgeTAM CoreML models for video tracking with persistent masks.
Shows both single-frame segmentation and multi-frame tracking workflows.

Author: Krish Mehta (https://github.com/DjKesu)
EdgeTAM Model: https://github.com/chuangg/EdgeTAM
"""

import argparse
import os
import sys

import coremltools as ct
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EdgeTAMVideoTracker:
    def __init__(self, model_dir="./coreml_models"):
        """Initialize EdgeTAM video tracker with CoreML models."""
        print("Loading EdgeTAM CoreML models...")
        self.image_encoder = ct.models.MLModel(
            f"{model_dir}/edgetam_image_encoder.mlpackage"
        )
        self.prompt_encoder = ct.models.MLModel(
            f"{model_dir}/edgetam_prompt_encoder.mlpackage"
        )
        self.mask_decoder = ct.models.MLModel(
            f"{model_dir}/edgetam_mask_decoder.mlpackage"
        )

        # Get positional encoding from PyTorch model if available
        try:
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.build_sam import _load_checkpoint

            device = torch.device("cpu")
            GlobalHydra.instance().clear()

            config_path = os.path.abspath("../sam2/configs/edgetam.yaml")
            config_dir = os.path.dirname(config_path)
            config_name = os.path.splitext(os.path.basename(config_path))[0]

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name)
                OmegaConf.resolve(cfg)
                model = instantiate(cfg.model, _recursive_=True)
                _load_checkpoint(model, "../checkpoints/edgetam.pt")
                model = model.to(device).eval()

                with torch.no_grad():
                    self.image_pe = (
                        model.sam_prompt_encoder.get_dense_pe().detach().numpy()
                    )
                print("Using proper positional encoding from PyTorch model")
        except:
            print("PyTorch model not available, using zeros for positional encoding")
            self.image_pe = np.zeros((1, 256, 64, 64), dtype=np.float32)

        self.current_embeddings = None
        self.tracking_points = []

    def encode_frame(self, frame):
        """Encode a video frame for tracking."""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame = frame.resize((1024, 1024), Image.BILINEAR)
        self.current_embeddings = self.image_encoder.predict({"image": frame})
        return self.current_embeddings

    def add_point(self, x, y, label=1):
        """Add a tracking point. Label: 1=positive, 0=negative."""
        self.tracking_points.append((x, y, label))

    def clear_points(self):
        """Clear all tracking points."""
        self.tracking_points = []

    def segment_current_frame(self, multimask=False):
        """Generate mask for current frame using tracking points."""
        if self.current_embeddings is None:
            raise ValueError("No frame encoded. Call encode_frame() first.")

        if not self.tracking_points:
            raise ValueError("No tracking points. Call add_point() first.")

        # Prepare point prompts
        max_points = 4
        point_coords = np.zeros((1, max_points, 2), dtype=np.float32)
        point_labels = np.full((1, max_points), -1, dtype=np.float32)

        for i, (x, y, label) in enumerate(self.tracking_points[:max_points]):
            point_coords[0, i] = [x, y]
            point_labels[0, i] = label

        # Encode prompts
        prompt_out = self.prompt_encoder.predict(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "boxes": np.zeros((1, 4), dtype=np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            }
        )

        # Generate mask
        mask_out = self.mask_decoder.predict(
            {
                "image_embeddings": self.current_embeddings["vision_features"],
                "image_pe": self.image_pe,
                "sparse_prompt_embeddings": prompt_out["sparse_embeddings"],
                "dense_prompt_embeddings": prompt_out["dense_embeddings"],
                "high_res_feat_0": self.current_embeddings["high_res_feat_0"],
                "high_res_feat_1": self.current_embeddings["high_res_feat_1"],
                "multimask_output": np.array(
                    [1.0 if multimask else 0.0], dtype=np.float32
                ),
            }
        )

        masks = mask_out["masks"]
        iou_scores = mask_out["iou_pred"]

        if multimask:
            best_idx = np.argmax(iou_scores)
            return masks[0, best_idx], iou_scores[0, best_idx]
        else:
            return masks[0, 0], iou_scores[0, 0]


def track_video_example(video_path=None, save_output=False):
    """Example: Track objects in a video stream."""
    tracker = EdgeTAMVideoTracker()

    # Initialize video capture (webcam or video file)
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)  # Use webcam
        print("Video tracking started. Click on objects to track them.")

    print("Press 'c' to clear points, 'q' to quit")

    # Setup video writer if saving output
    video_writer = None
    if save_output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "tracked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert from display coordinates to model coordinates
            model_x = int(x * 1024 / frame.shape[1])
            model_y = int(y * 1024 / frame.shape[0])
            tracker.add_point(model_x, model_y, label=1)
            print(f"Added tracking point at ({model_x}, {model_y})")

    cv2.namedWindow("EdgeTAM Video Tracking")
    cv2.setMouseCallback("EdgeTAM Video Tracking", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame
        tracker.encode_frame(frame)

        # Generate mask if we have tracking points
        if tracker.tracking_points:
            try:
                mask, iou = tracker.segment_current_frame(multimask=False)

                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.0).astype(np.uint8) * 255

                # Create colored overlay
                overlay = frame.copy()
                overlay[mask_binary > 0] = [0, 255, 0]  # Green overlay
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                # Show IoU score
                cv2.putText(
                    frame,
                    f"IoU: {iou:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            except Exception as e:
                print(f"Tracking error: {e}")

        # Draw tracking points
        for i, (x, y, label) in enumerate(tracker.tracking_points):
            display_x = int(x * frame.shape[1] / 1024)
            display_y = int(y * frame.shape[0] / 1024)
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(frame, (display_x, display_y), 5, color, -1)

        cv2.imshow("EdgeTAM Video Tracking", frame)

        # Write frame to output video if saving
        if video_writer is not None:
            video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            tracker.clear_points()
            print("Cleared all tracking points")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {output_path}")
    cv2.destroyAllWindows()


def segment_image_example():
    """Example: Segment a single image with point prompt."""
    tracker = EdgeTAMVideoTracker()

    # Create test image
    test_image = Image.new("RGB", (1024, 1024), color="lightblue")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(test_image)
    draw.ellipse([200, 200, 400, 400], fill="red")
    draw.rectangle([600, 300, 800, 500], fill="green")

    # Encode image
    tracker.encode_frame(test_image)

    # Add point in red circle
    tracker.add_point(300, 300, label=1)

    # Generate mask
    mask, iou = tracker.segment_current_frame(multimask=False)

    print(f"Segmentation complete! IoU: {iou:.3f}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask coverage: {(mask > 0.0).sum() / (1024 * 1024) * 100:.1f}%")

    return mask


def demo_with_video(video_path=None):
    """Demo tracking with example video.

    Args:
        video_path: Path to video file. If None, defaults to "../examples/04_coffee.mp4"
    """
    if video_path is None:
        video_path = "../examples/04_coffee.mp4"

    print("EdgeTAM CoreML Video Demo")
    print("========================")
    print(f"Processing: {video_path}")
    print("Will automatically track object at center of frame")
    print("Output will be saved as 'tracked_output.mp4'")
    print()

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        examples_dir = (
            os.path.dirname(video_path)
            if video_path != "../examples/04_coffee.mp4"
            else "../examples/"
        )
        if os.path.exists(examples_dir):
            print("Available videos:")
            for f in os.listdir(examples_dir):
                if f.endswith((".mp4", ".avi", ".mov")):
                    print(f"  {os.path.join(examples_dir, f)}")
        return

    tracker = EdgeTAMVideoTracker()
    cap = cv2.VideoCapture(video_path)

    # Setup video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving output to: {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame
        tracker.encode_frame(frame)

        # Add tracking point on first frame
        if frame_count == 0:
            # Add point on the coffee cup (center-right area where cup usually appears)
            tracker.add_point(690, 600, label=1)  # Coffee cup position
            print(f"Added tracking point at (690, 600) - coffee cup")

        # Generate mask if we have tracking points
        if tracker.tracking_points:
            try:
                mask, iou = tracker.segment_current_frame(multimask=False)
                print(f"Frame {frame_count}: IoU {iou:.3f}")

                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.0).astype(np.uint8) * 255

                # Create colored overlay
                overlay = frame.copy()
                overlay[mask_binary > 0] = [0, 255, 0]  # Green overlay
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                # Show IoU score
                cv2.putText(
                    frame,
                    f"IoU: {iou:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            except Exception as e:
                print(f"Frame {frame_count} error: {e}")

        # Draw tracking points
        for i, (x, y, label) in enumerate(tracker.tracking_points):
            display_x = int(x * frame.shape[1] / 1024)
            display_y = int(y * frame.shape[0] / 1024)
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(frame, (display_x, display_y), 5, color, -1)

        # Write frame
        video_writer.write(frame)
        frame_count += 1

        # Process only first 100 frames for demo
        if frame_count >= 100:
            break

    cap.release()
    video_writer.release()
    print(f"Processed {frame_count} frames")
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EdgeTAM CoreML Video Tracking Example"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (default: ../examples/04_coffee.mp4)",
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["segment", "track", "demo"],
        default="demo",
        help="Which example to run: segment, track, or demo (default: demo)",
    )

    args = parser.parse_args()

    print("EdgeTAM CoreML Video Tracking Example")
    print("=====================================")
    print()
    print("Available examples:")
    print("1. segment_image_example() - Single image segmentation")
    print("2. track_video_example() - Real-time video tracking")
    print("3. demo_with_video() - Demo with example video")
    print()

    # Run demo with video by default
    try:
        if args.example == "segment":
            segment_image_example()
        elif args.example == "track":
            track_video_example(video_path=args.video)
        else:  # demo
            demo_with_video(video_path=args.video)
    except FileNotFoundError:
        print("Error: CoreML models not found!")
        print("Please run the export script first:")
        print(
            "  python coreml/export_to_coreml.py --sam2_cfg sam2/configs/edgetam.yaml --sam2_checkpoint checkpoints/edgetam.pt"
        )
