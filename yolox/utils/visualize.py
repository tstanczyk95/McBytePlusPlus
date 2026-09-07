#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
from typing import Sequence
import torch

__all__ = ["vis"]

def id_to_color_bgr(oid: int) -> np.ndarray:
    """
    Deterministic pseudo-random BGR color for a given object id.
    Always returns the same BGR for the same oid.
    """
    # simple hash-based color
    r = (37 * oid + 17)  % 256
    g = (17 * oid + 59)  % 256
    b = (29 * oid + 133) % 256
    # OpenCV uses BGR order
    return np.array([b, g, r], dtype=np.uint8)



def overlay_edgetam_masks_on_frame_bgr(
    frame_bgr: np.ndarray,
    obj_ids: Sequence[int],
    mask_logits: torch.Tensor,
    alpha: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """
    Overlay EdgeTAM masks on an OpenCV BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Image array in BGR format, shape (H, W, 3), dtype uint8.
    obj_ids : sequence of int
        Object IDs for this frame (as returned by EdgeTAM).
    mask_logits : torch.Tensor or np.ndarray
        Tensor of shape (N, H, W) or (N, 1, H, W) with mask logits.
    alpha : float
        Blend factor for mask color vs original image (0..1).
    copy : bool
        If True, work on a copy of the frame. If False, modify in-place.

    Returns
    -------
    out_bgr : np.ndarray
        Frame with colored masks overlayed (BGR, uint8).
    """
    if copy:
        arr = frame_bgr.copy()
    else:
        arr = frame_bgr  # in-place

    H, W = arr.shape[:2]

    # Ensure mask_logits is on CPU and numpy
    if isinstance(mask_logits, torch.Tensor):
        mask_np = mask_logits.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_logits)

    # Expected shapes: (N, H, W) or (N, 1, H, W)
    if mask_np.ndim == 4 and mask_np.shape[1] == 1:
        # (N,1,H,W) -> (N,H,W)
        mask_np = mask_np[:, 0, :, :]
    elif mask_np.ndim != 3:
        raise ValueError(f"Unexpected mask_logits shape {mask_np.shape}, "
                         "expected (N,H,W) or (N,1,H,W)")

    N = mask_np.shape[0]
    if N != len(obj_ids):
        raise ValueError(f"mask count ({N}) != len(obj_ids) ({len(obj_ids)})")

    for i, oid in enumerate(obj_ids):
        m = mask_np[i]

        # Convert logits to boolean mask; threshold at 0
        if m.dtype != bool:
            m = m > 0.0

        if m.shape != (H, W):
            raise ValueError(
                f"Mask shape {m.shape} does not match frame shape {(H, W)}"
            )
        if not m.any():
            continue  # skip empty masks

        color = id_to_color_bgr(int(oid))  # (3,)

        # Blend in-place: arr[m] is (num_pixels, 3)
        arr_m = arr[m]
        arr[m] = ((1.0 - alpha) * arr_m + alpha * color).astype(np.uint8)

    return arr


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking_basic(image, tlwhs, obj_ids, frame_id=0, ids2=None, prediction_mask=None):
    im = np.ascontiguousarray(np.copy(image))
 
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    if prediction_mask is not None:
        mask_ids = prediction_mask["mask_ids"]
        mask_logits = prediction_mask["mask_logits"]
        im = overlay_edgetam_masks_on_frame_bgr(im, mask_ids, mask_logits, copy=False)

    # Online tracklets
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    return im, None, None


def plot_tracking(image, tlwhs, obj_ids, frame_id=0, ids2=None, prediction_mask=None, det_dict=None, considered_online_tlwhs_before_correction=[], considered_online_ids_of_tracks_before_correction=[]): 
    im = np.ascontiguousarray(np.copy(image))
    im_with_dets = np.ascontiguousarray(np.copy(image)) # For visualizing detections
    if len(considered_online_tlwhs_before_correction) > 0:
        im_tracks_before_update =  np.ascontiguousarray(np.copy(image)) # For visualizing all considered tracklets before the Kalman Filter's correction (update)
    else:
        im_tracks_before_update = None
        
    text_scale = 2
    text_thickness = 2
    line_thickness = 3


    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    cv2.putText(im_with_dets, 'frame: %d' % (frame_id),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    if len(considered_online_tlwhs_before_correction) > 0:
        cv2.putText(im_tracks_before_update, 'frame: %d num: %d' % (frame_id, len(considered_online_tlwhs_before_correction)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    
    # Put segmentation masks on the image
    if prediction_mask is not None:
        mask_ids = prediction_mask["mask_ids"]
        mask_logits = prediction_mask["mask_logits"]
        im = overlay_edgetam_masks_on_frame_bgr(im, mask_ids, mask_logits, copy=False)

    # Online tracklets
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    # All considered tracklets before the Kalman Filter's correction (update)
    if len(considered_online_tlwhs_before_correction) > 0:
        for i, tlwh in enumerate(considered_online_tlwhs_before_correction):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(considered_online_ids_of_tracks_before_correction[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = get_color(abs(obj_id))
            cv2.rectangle(im_tracks_before_update, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im_tracks_before_update, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
    


    # Detections (5 categories)
    for i, det in enumerate(det_dict['assoc1']):
        tlwh = det.tlwh
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = i
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (0, 255, 0) # BGR, green
        cv2.rectangle(im_with_dets, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im_with_dets, id_text, (intbox[0], intbox[1] + 25), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0),
                    thickness=text_thickness)

    for i, det in enumerate(det_dict['assoc2']):
        tlwh = det.tlwh
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = i
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (255, 0, 0) # BGR, blue
        cv2.rectangle(im_with_dets, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im_with_dets, id_text, (intbox[0], intbox[1] + 50), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0),
                    thickness=text_thickness)
        
    for i, det in enumerate(det_dict['assoc3']):
        tlwh = det.tlwh
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = i
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (0, 255, 255) # BGR, yellow
        cv2.rectangle(im_with_dets, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im_with_dets, id_text, (intbox[0], intbox[1] + 75), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                    thickness=text_thickness)
        
    for i, det in enumerate(det_dict['init_acc']):
        tlwh = det.tlwh
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = i
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (0, 165, 255) # BGR, orange
        cv2.rectangle(im_with_dets, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im_with_dets, id_text, (intbox[0], intbox[1] + 100), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 165, 255),
                    thickness=text_thickness)
        
    for i, det in enumerate(det_dict['init_rej']):
        tlwh = det.tlwh
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = i
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (0, 0, 255) # BGR, red
        cv2.rectangle(im_with_dets, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im_with_dets, id_text, (intbox[0], intbox[1] + 125), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)


    return im, im_with_dets, im_tracks_before_update


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
