import numpy as np
np.set_printoptions(threshold=np.inf)

from collections import deque
import os.path as osp
import copy
import copy

from .kalman_filter import KalmanFilter
from yolox.tracker import matching as matching
from yolox.tracker.gmc import GMC
from .basetrack import BaseTrack, TrackState

import cv2
from PIL import Image

# re-ID related imports
try:
    import torch
    import torchvision.transforms as T
    from torchreid.utils import FeatureExtractor
    _HAS_REID = True
except ImportError:
    _HAS_REID = False

### Constants ### 
MIN_MASK_AVG_CONF = 0.6
MIN_MM1 = 0.9
MIN_MM2 = 0.05

MAX_COST_1ST_ASSOC_STEP = 0.9
MAX_COST_2ND_ASSOC_STEP = 0.5
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.7

def _warp_looks_safe(H):
    if H is None:
        return False
    H = np.asarray(H, dtype=np.float32)
    if H.shape != (2, 3):
        return False
    if not np.isfinite(H).all():
        return False
    R = H[:2, :2]
    sx = float(np.linalg.norm(R[:, 0]))
    sy = float(np.linalg.norm(R[:, 1]))
    # super conservative: only reject collapses/explosions
    if sx < 0.3 or sx > 3.0:
        return False
    if sy < 0.3 or sy > 3.0:
        return False
    return True

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.last_det_tlwh = tlwh

        ### re-ID related state ###
        # deque of np.ndarray feature vectors (L2-normalized)
        self.reid_feats = deque(maxlen=50)     # you can tune maxlen
        # cached mean of all features in reid_feats (np.ndarray or None)
        self.reid_feat_avg = None
        # last frame at which we updated re-ID (to avoid over-sampling)
        self.reid_last_frame = -1

        # --- Export-ID layer ---
        self.export_id = None           # set on activate
        self.export_id_final = False    # once true, we stop trying to reconnect it
        self.reid_num_updates = 0       # how many frames contributed re-id features

    def add_reid_feature(self, feat: np.ndarray, frame_id: int):
        """
        Add a new L2-normalized re-ID feature vector and update running mean.
        feat: np.ndarray of shape (D,), already L2-normalized
        """
        if feat is None:
            return
        self.reid_feats.append(feat)

        # simple mean over all stored features
        feats_stack = np.stack(self.reid_feats, axis=0)  # (N, D)
        # keep them normalized (each row unit norm)
        norms = np.linalg.norm(feats_stack, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        feats_stack = feats_stack / norms
        self.reid_feat_avg = feats_stack.mean(axis=0)
        # re-normalize the mean as well (useful later for cosine similarity)
        n = np.linalg.norm(self.reid_feat_avg)
        if n > 0:
            self.reid_feat_avg = self.reid_feat_avg / n

        self.reid_last_frame = frame_id
        self.reid_num_updates += 1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
    

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.export_id = self.track_id
        # Tracks born at the very first frame cannot be a "re-appearance" within this sequence,
        # so don't treat them as provisional (no buffering, no -1 in vis).
        self.export_id_final = (frame_id == 1)
        self.reid_num_updates = 0

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        # keep export_id of the existing track
        if self.export_id is None:
            self.export_id = self.track_id

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)


    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class McByteLogger(object):
    def __init__(self, log_file_path):
        self.file = open(log_file_path, "w")
        np.set_printoptions(linewidth=1000)

    def __del__(self):
        self.file.close()

    def log_info(self):
        pass

    def log_frame_no(self, frame_no):
        self.file.write("\n\n= = = = = = Frame number: " + str(frame_no) + " = = = = =\n\n")

    def log_mask_info(self, tracklet_mask_dict):
        self.file.write("tracklet_id -> mask_number:\n")
        if tracklet_mask_dict is None:
            self.file.write("< tracklet_mask_dict is None, probably frame(s) before creating the masks with SAM >")
        else:
            for k, v in tracklet_mask_dict.items():
                self.file.write(str(k) + " -> " + str(v) + ", ")
        self.file.write("\n\n") 

    def log_dists(self, dists, mask_match_included, which_association, frame_no):
        self.file.write("Association step: " + str(which_association) + " (frame " + str(frame_no) + ")" + "\nMask match included: " + str(mask_match_included) + "\n\n")
        self.file.write(str(dists) + "\n\n")

    def log_matches(self, matches, u_track, u_detection, strack_pool_ids):
        self.file.write("matrix row -> tracklet_id:\n")
        for i in range(len(strack_pool_ids)):
            self.file.write(str(i) + " -> " + str(strack_pool_ids[i]) + ", ")
        self.file.write("\n\n")        

        self.file.write("matches [row column] [track det]:\n")# + str(matches) + "\n")
        for match in matches:
            self.file.write(str(match) + "\n")
        self.file.write("u_track:\n" + str(u_track) + "\n")
        self.file.write("u_detection:\n" + str(u_detection) + "\n")
        
    def log_det_conf_scores(self, detections):
        self.file.write("Detection confidence scores:\n")
        for i, det in enumerate(detections):
            self.file.write(str(i) + " : " + str(np.round(det.score, decimals=2)) + "\t")
        self.file.write("\n\n")  
        self.file.write("- - - - - - - - - -\n\n")

    def log_local_update_trackets_ids(self, activated, refind, lost, removed):
        self.file.write("activated_stracks: ")
        for track in activated:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nrefind_stracks: ")
        for track in refind:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nlost_stracks: ")
        for track in lost:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nremoved_stracks: ")
        for track in removed:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\n\n")

    def log_state_tracklets_ids(self, tracked, lost, removed):
        self.file.write("self.track_stracks: ")
        for track in tracked:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nself.lost_stracks: ")
        for track in lost:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nself.removed_stracks: ")
        for track in removed:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\n\n")


class ReIDManager(object):
    """
    GTA-link style re-ID wrapper:
      - osnet_x1_0 backbone
      - same transforms as generate_tracklets.py
      - L2-normalized embeddings
      GTA-link: https://github.com/sjc042/gta-link
    """

    def __init__(self, model_path: str, device: str = "cuda", max_feats_per_track: int = 50):
        self.enabled = _HAS_REID and (model_path is not None)
        if not self.enabled:
            print("[ReIDManager] Re-ID disabled (missing torchreid or model_path).")
            return

        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        print(f"[ReIDManager] Initializing on device: {self.device}")

        self.transforms = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=model_path,
            device=self.device,
        )

        self.max_feats_per_track = max_feats_per_track

    def is_enabled(self) -> bool:
        return self.enabled

    def _crop_box_from_tlwh(self, frame_img, tlwh):
        """
        tlwh: [x, y, w, h] in image coordinates
        Returns cropped RGB image as PIL.Image or None if invalid.
        """
        if frame_img is None:
            return None

        x, y, w, h = tlwh
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(frame_img.shape[1], x + w))
        y2 = int(min(frame_img.shape[0], y + h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_img[y1:y2, x1:x2, :]
        if crop.size == 0:
            return None

        # BGR -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        return pil_img

    def compute_feature_from_tlwh(self, frame_img, tlwh):
        """
        Returns a single L2-normalized feature (np.ndarray, shape (D,))
        or None if something went wrong.
        """
        if not self.enabled:
            return None

        pil_img = self._crop_box_from_tlwh(frame_img, tlwh)
        if pil_img is None:
            return None

        with torch.no_grad():
            tensor = self.transforms(pil_img).unsqueeze(0)  # (1,3,256,128)
            feats = self.extractor(tensor.to(self.device))  # (1, D)
            feats_np = feats.cpu().detach().numpy().reshape(-1)  # (D,)

        # L2-normalize
        n = np.linalg.norm(feats_np)
        if n == 0:
            return None
        feats_np = feats_np / n
        return feats_np


   
class McBytePlusPlusTracker(object):
    def __init__(self, args, save_folder, frame_rate=30, enable_logging=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # For camera motion compensation
        self.gmc = GMC(method=args.cmc_method, downscale=args.cmc_downscale, verbose=None) if args.cmc_method != "none" else None
        self.gmc_interval = args.cmc_interval
        self.last_warp = np.eye(2, 3, dtype=np.float32)


        if enable_logging:
            self.save_folder = save_folder
            log_file_path = osp.join(self.save_folder, "logging_info.txt")
            self.logger = McByteLogger(log_file_path)
        else:
            self.logger = None

        ### re-ID manager ###
        # Expect args.reid_model_path, args.reid_device, args.reid_max_feats (optional)
        reid_model_path = getattr(args, "reid_model_path", None)
        reid_device = getattr(args, "reid_device", "cuda")
        reid_max_feats = getattr(args, "reid_max_feats", 50)

        self.reid_manager = ReIDManager(
            model_path=reid_model_path,
            device=reid_device,
            max_feats_per_track=reid_max_feats,
        )

        # Heuristics for when to compute re-ID features
        self.reid_min_det_score = getattr(args, "reid_min_det_score", 0.6)
        self.reid_min_box_area = getattr(args, "reid_min_box_area", 32 * 32)
        self.reid_min_box_w     = getattr(args, "reid_min_box_w", 25)
        self.reid_min_box_h     = getattr(args, "reid_min_box_h", 50)
        self.reid_border_margin = getattr(args, "reid_border_margin", 4)
        self.reid_min_frame_gap = getattr(args, "reid_min_frame_gap", 5)  # frames between feature updates

        # export_id -> dict(feat, end_frame, last_tlwh)
        self.reid_identity_bank = {}

        # reconnection params
        self.reid_reconnect_enabled = getattr(args, "reid_reconnect_enabled", True)
        self.reid_sim_thresh        = getattr(args, "reid_sim_thresh", 0.7)  # you will tune 0.70–0.75
        self.reid_new_max_len       = getattr(args, "reid_new_max_len", 10)   # treat as "new tracklet" window
        self.reid_min_updates       = getattr(args, "reid_min_updates", 2)    # number of good re-ID updates required

    def _stash_identity_if_ready(self, track: STrack):
        """
        Store terminated tracklet identity into memory.
        We only stash if it has a descriptor.
        """
        if getattr(track, "export_id", None) is None:
            return
        if track.reid_feat_avg is None:
            return

        # Don’t store weak identities
        if getattr(track, "reid_num_updates", 0) < self.reid_min_updates:
            return

        eid = track.export_id
        self.reid_identity_bank[eid] = {
            "feat": track.reid_feat_avg.copy(),   # already L2-normalized
            "end_frame": track.end_frame,
            "last_tlwh": track.last_det_tlwh,
        }

    def _tlwh_center(self, tlwh):
        x, y, w, h = tlwh
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=float)

    def _cos_sim(self, a, b):
        if a is None or b is None:
            return None
        return float(np.dot(a, b))  # both L2-normalized

    def _get_new_reid_tracks(self):
        """
        Tracks eligible for reconnection.
        """
        res = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                continue
            if t.state != TrackState.Tracked:
                continue
            if getattr(t, "export_id_final", False):
                continue
            if t.reid_feat_avg is None:
                continue
            if getattr(t, "reid_num_updates", 0) < self.reid_min_updates:
                continue

            # "newness": age since birth
            if (self.frame_id - t.start_frame) > self.reid_new_max_len:
                continue

            # only attempt reconnection while still provisional
            if t.export_id != t.track_id:
                continue

            res.append(t)
        return res

    def _get_identity_candidates(self):
        """
        Identity candidates from memory.
        """
        return list(self.reid_identity_bank.items())  # (export_id, info)
    
    def _get_active_export_ids(self):
        active = set()
        for t in self.tracked_stracks:
            if t.is_activated and t.state == TrackState.Tracked:
                eid = getattr(t, "export_id", None)
                if eid is not None:
                    active.add(eid)
        return active

    def _get_reserved_export_ids(self):
        reserved = set()
        for lst in [self.tracked_stracks, self.lost_stracks]:
            for t in lst:
                eid = getattr(t, "export_id", None)
                if eid is not None:
                    reserved.add(eid)
        return reserved
    

    def _ensure_unique_export_ids(self):
        """
        Hard guarantee: among active Tracked tracks, export_id must be unique.
        If duplicates exist, keep the "best" owner for a duplicated export_id and
        revert the others back to their own unique IDs (track_id).

        This prevents writing the same ID twice in a single frame.
        """
        # Only consider tracks that can be output (Tracked + activated)
        active = []
        for t in self.tracked_stracks:
            if not getattr(t, "is_activated", False):
                continue
            if getattr(t, "state", None) != TrackState.Tracked:
                continue
            active.append(t)

        if not active:
            return

        def keep_key(t):
            # Prefer IDs that are "final" (reconnected), then longer tracklet, then higher score
            return (
                1 if getattr(t, "export_id_final", False) else 0,
                int(getattr(t, "tracklet_len", 0)),
                float(getattr(t, "score", 0.0)),
            )

        eid_owner = {} # export_id -> track (current best owner)
        dup_eids = set()

        for t in active:
            eid = getattr(t, "export_id", None)
            if eid is None:
                # Should not happen after activate(), but keep it safe
                t.export_id = t.track_id
                t.export_id_final = False
                eid = t.export_id

            if eid not in eid_owner:
                eid_owner[eid] = t
                continue

            # Conflict: two tracks have the same export_id
            dup_eids.add(eid)
            cur = eid_owner[eid]

            if keep_key(t) > keep_key(cur):
                # New one is better -> demote previous owner
                cur.export_id = cur.track_id
                cur.export_id_final = False
                eid_owner[eid] = t
            else:
                # Demote this one
                t.export_id = t.track_id
                t.export_id_final = False

        if dup_eids:
            print(f"[WARN][export_id] frame={self.frame_id} fixed duplicate export_id(s): {sorted(dup_eids)}")


    def _finalize_old_provisional_export_ids(self):
        """
        Mark 'young new tracks' as finalized once they are older than reid_new_max_len,
        i.e. we stop waiting for a reconnection. This means:
        - if they never reconnected: they become a confirmed new identity
        - export_id stays == track_id, but export_id_final becomes True
        This is crucial for output buffering (demo) to know when to flush.
        """
        for t in self.tracked_stracks:
            if not getattr(t, "is_activated", False):
                continue
            if getattr(t, "state", None) != TrackState.Tracked:
                continue

            if getattr(t, "export_id_final", False):
                continue

            # If it already reconnected (export_id != track_id) but flag wasn't set for some reason, finalize.
            if getattr(t, "export_id", t.track_id) != t.track_id:
                t.export_id_final = True
                continue

            # Confirm as "new / unrelated" once it exceeds the reconnection window.
            if (self.frame_id - t.start_frame) > self.reid_new_max_len:
                t.export_id_final = True


    def try_reconnect_new_tracks(self):
        if not self.reid_reconnect_enabled:
            return
        if not self.reid_manager.is_enabled():
            return
        if len(self.reid_identity_bank) == 0:
            return

        reserved_ids = self._get_reserved_export_ids()

        new_tracks = self._get_new_reid_tracks()
        if not new_tracks:
            return

        identities = self._get_identity_candidates()
        if not identities:
            return

        # Build similarity matrix S: (N_new, N_id)
        N = len(new_tracks)
        M = len(identities)
        S = np.full((N, M), -1.0, dtype=float)

        for i, nt in enumerate(new_tracks):
            for j, (eid, info) in enumerate(identities):
                # DO NOT connect to an ID already in use (prevents duplicates)
                if eid in reserved_ids:
                    continue

                sim = self._cos_sim(nt.reid_feat_avg, info["feat"])
                if sim is None:
                    continue
                if sim < self.reid_sim_thresh:
                    continue

                S[i, j] = sim

        # If no valid candidates
        if np.all(S < 0):
            return

        # best identity for each new track
        best_j_for_i = np.argmax(S, axis=1)
        best_sim_for_i = S[np.arange(N), best_j_for_i]

        # best new track for each identity
        best_i_for_j = np.argmax(S, axis=0)

        # Apply mutual nearest neighbor + threshold
        for i in range(N):
            j = int(best_j_for_i[i])
            sim = float(best_sim_for_i[i])
            if sim < self.reid_sim_thresh:
                continue
            if S[i, j] < 0:
                continue
            if best_i_for_j[j] != i:
                continue

            eid, info = identities[j]

            if eid in reserved_ids:
                continue

            # Assign export_id (variant B)
            nt = new_tracks[i]
            old_export = nt.export_id
            nt.export_id = eid
            nt.export_id_final = True  # stop further reconnect attempts for this tracklet
            reserved_ids.add(eid)

            print(
                f"[ReID RECONNECT] frame={self.frame_id} "
                f"new(track_id={nt.track_id}, age={self.frame_id-nt.start_frame}, old_export={old_export}) -> export_id={eid} "
                f"sim={sim:.3f}"
            )

            # after successful reconnection
            del self.reid_identity_bank[eid]


    def _update_reid_features_for_tracks(self, frame_img):
        """
        Update re-ID features for active tracks on the current frame.
        Uses rough heuristics:
          - only for is_activated & Tracked tracks
          - only if detection score >= reid_min_det_score
          - only if bbox area >= reid_min_box_area
          - only every reid_min_frame_gap frames per track
          - optionally: only if mask is present and confident (approx non-occluded) #TODO - discard this one
        """
        if not self.reid_manager.is_enabled():
            return

        if frame_img is None:
            return

        img_h, img_w = frame_img.shape[:2]
        m = self.reid_border_margin

        for track in self.tracked_stracks:
            if not track.is_activated:
                continue
            if track.state != TrackState.Tracked:
                continue

            # Don't over-sample re-ID (only update every N frames)
            if track.reid_last_frame >= 0:
                if (self.frame_id - track.reid_last_frame) < self.reid_min_frame_gap:
                    continue

            tlwh = track.last_det_tlwh
            x, y, w, h = tlwh
            area = w * h
            if area < self.reid_min_box_area:
                continue
            if w < self.reid_min_box_w or h < self.reid_min_box_h:
                continue
            if track.score < self.reid_min_det_score:
                continue

            # Border-touch skip (likely truncated)
            if (x <= m) or (y <= m) or ((x + w) >= (img_w - m)) or ((y + h) >= (img_h - m)):
                continue

            # Compute re-ID feature and attach to the track
            feat = self.reid_manager.compute_feature_from_tlwh(frame_img, tlwh)
            if feat is not None:
                track.add_reid_feature(feat, frame_id=self.frame_id)
                track.reid_num_updates += 1


    def _is_new_tracklet(self, track):
        # Conservative definition: very young track
        return (
            track.is_activated
            and track.state == TrackState.Tracked
            and track.tracklet_len <= 5    # tune later (e.g. 3–5)
            and track.reid_feat_avg is not None
        )
    

    def _get_existing_tracklets_for_reid(self):
        """
        Returns a list of tracks that are candidates for re-ID matching.
        """
        candidates = []
        for t in self.lost_stracks + self.removed_stracks:
            if t.reid_feat_avg is None:
                continue
            candidates.append(t)
        return candidates


    def debug_print_reid_similarities(self, top_k=5, min_sim=0.0):
        """
        Print cosine similarities between new tracklets and existing (terminated) tracklets.
        Diagnostic only — no ID changes.
        """

        new_tracks = [t for t in self.tracked_stracks if self._is_new_tracklet(t)]
        old_tracks = self._get_existing_tracklets_for_reid()

        if not new_tracks or not old_tracks:
            return

        print("\n[ReID DEBUG] Cosine similarities (new → old):")

        for new_t in new_tracks:
            print(f"\n--- New track {new_t.track_id} (len={new_t.tracklet_len}) ---")

            sims = []
            for old_t in old_tracks:
                sim = cosine_similarity_np(new_t.reid_feat_avg, old_t.reid_feat_avg)
                if sim is None:
                    continue
                if sim >= min_sim:
                    sims.append((old_t.track_id, sim, old_t))

            if not sims:
                print("  (no comparable old tracks)")
                continue

            # sort descending by similarity
            sims.sort(key=lambda x: x[1], reverse=True)

            for old_id, sim, old_t in sims[:top_k]:
                print(
                    f"  vs old track {old_id:4d} | sim={sim:.4f} "
                    f"| last_frame={old_t.end_frame}"
                )
 
    
    def conditioned_assignment(self, dists, max_cost, strack_pool, detections, prediction_mask, tracklet_mask_dict, img_info):
        dists_cp = np.copy(dists)

        if prediction_mask is not None:
            mask_ids = prediction_mask["mask_ids"]
            mask_logits = prediction_mask["mask_logits"]

        # Go through each entry in the dists matrix
        for i in range(dists_cp.shape[0]):
            for j in range(dists_cp.shape[1]):
                if dists[i,j] <= max_cost:
                    # Check if there are other entries in the same row or column (=other tracklets or detections) meeting this condition
                    if not (sum(dists[i, :] <= max_cost) > 1 or sum(dists[:, j] <= max_cost) > 1):
                        # NOT the case, then it's a clear match
                        # Set all the entries in this row and all the entries in this column to 1 (high cost). Yet ofc, keep the current entry with its original value.
                        # +10 for debugging purposes and analysis. It doesn't change output compared with setting it to 1 as all the entries above max_cost (max_cost < 1) 
                        # will be rejected from matching
                        dists_cp[i, :] += 10
                        dists_cp[:, j] += 10
                        dists_cp[i,j] = dists[i,j]
                    
                    else:
                        if prediction_mask is None: 
                            continue
                        
                        # It is the case. Update the dists matrix based on the other cue(s)
                        strack = strack_pool[i]
                        det = detections[j]

                        # If there exists a mask for this tracklet
                        strack_id = strack.track_id
                        if strack_id in tracklet_mask_dict.keys():
                            strack_mask_id = tracklet_mask_dict[strack_id]
                            mask_id_position = mask_ids.index(strack_mask_id)

                            # If this mask is present at the scene right now
                            mask_all_valid_pixels = (mask_logits[mask_id_position][0] > 0.0).sum()
                            if mask_all_valid_pixels > 0:

                                # NOT INCLUDED FOR EDGETAM IN MCBYTE++. EdgeTAM doesn't provide such information.
                                # if mask_avg_prob_dict[strack_mask_id] >= MIN_MASK_AVG_CONF:
                                    img_h, img_w = img_info[0], img_info[1]
                                    
                                    # Get detection coordinates and prepare them for computing mm1 and mm2
                                    x, y, w, h = det.tlwh

                                    x = int(x)
                                    if x < 0: 
                                        x = 0

                                    y = int(y)
                                    if y < 0:
                                        y = 0
        
                                    w = int(w)
                                    if x + w > img_w:
                                        hor_bound = img_w
                                    else:
                                        hor_bound = x + w

                                    h = int(h)
                                    if y + h > img_h:
                                        ver_bound = img_h
                                    else:
                                        ver_bound = y + h

                                    # Compute mm1 and mm2 (mask coverage of the bounding box ratios)
                                    # mm1 ("mc" in the paper): "the bounding box coverage of the mask" - the ratio of the number of the mask pixels in the bounding box to the number of all the mask pixels currently present at the scene
                                    # mm2 ("mf" in the paper): "the mask fill ratio of the bounding box" - the ratio of the number of the mask pixels in the bounding box to the number all the bounding box pixels
                                    
                                    mask_pixels_in_bb = (mask_logits[mask_id_position][0][y:ver_bound, x:hor_bound] > 0.0).sum()

                                    mask_match_opt_1 = mask_pixels_in_bb / mask_all_valid_pixels
                                    mask_match_opt_2 = mask_pixels_in_bb / ((ver_bound-y) * (hor_bound-x))

                                    # If mask is occupying at least certain part (percentage) of the bounding box
                                    if mask_match_opt_2 >= MIN_MM2:
                                        # If there is quite some of the mask present outside the bounbding box, then exclude this entry from optimization
                                        if mask_match_opt_1 < MIN_MM1:
                                            continue
                                        else:
                                            # Incorporate the mask cue information
                                            dists_cp[i,j] -= mask_match_opt_2

        # All dists_cp entries updated (when relevant) at this stage.
        # Get the matches with the Hungarian matching algorithm
        matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)

        return matches, u_track, u_detection, dists_cp


    def update(self, output_results, img_info, img_size, prediction_mask, tracklet_mask_dict, frame_img, vis_type, dets_from_file=False):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        assoc1_dets = []
        assoc2_dets = []
        assoc3_dets = []
        init_track_dets_acc = []
        init_track_dets_rej = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        
        if not dets_from_file:
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if self.logger is not None:
            self.logger.log_frame_no(self.frame_id)
            self.logger.log_state_tracklets_ids(self.tracked_stracks, self.lost_stracks, self.removed_stracks)
            self.logger.log_mask_info(tracklet_mask_dict)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []


        ''' Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        if self.gmc is not None:
            # Fix camera motion 
            try:
                if self.frame_id % self.gmc_interval == 0:
                    warp = self.gmc.apply(frame_img, dets)
                    # Only store warp if it looks sane
                    if _warp_looks_safe(warp):
                        self.last_warp = warp
                    else:
                        warp = None  # force skip this frame (or fall back below)
                else:
                    warp = self.last_warp

                if warp is not None:
                    STrack.multi_gmc(strack_pool, warp)
                    STrack.multi_gmc(unconfirmed, warp)
            except Exception:
                self.last_warp = None
                print("[Frame {}] Internal error while trying to apply the GMC, thus skipping camera motion compensation".format(str(self.frame_id)))
                pass

        # Do visualize all considered tracklets before KF correction (update):
        if vis_type == 'full':
            strack_pool_before_correction = copy.deepcopy(strack_pool)
            unconfirmed_before_correction = copy.deepcopy(unconfirmed)
            all_considered_tracklets_before_correction = joint_stracks(strack_pool_before_correction, unconfirmed_before_correction)
        else:
        # Do not:
            all_considered_tracklets_before_correction = None
        
        dists = matching.iou_distance(strack_pool, detections)
        # The buffered-IoU variant 1/3
        # dists = matching.buffered_iou_distance(strack_pool, detections, 0.3)
        
        dists = matching.fuse_score(dists, detections)
        
        if vis_type == 'full':
            assoc1_dets = [det for det in detections]  # For detection visualization (1/5)
        if self.logger is not None:
            self.logger.log_dists(dists, mask_match_included=False, which_association=1, frame_no=self.frame_id)
                                     
        matches, u_track, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_1ST_ASSOC_STEP, strack_pool, detections, prediction_mask, tracklet_mask_dict, img_info)
        if self.logger is not None:
            self.logger.log_dists(dists_cp, mask_match_included=True, which_association=1, frame_no=self.frame_id)
        
        if self.logger is not None:
            strack_pool_ids = [s.track_id for s in strack_pool]
            self.logger.log_matches(matches, u_track, u_detection, strack_pool_ids)
            self.logger.log_det_conf_scores(detections)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        ''' Step 3: Second association, with low score detection boxes'''
        # association of the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
            # detections_second = [] # quick change for excluding the whole second association step
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # The buffered-IoU variant 2/3
        # dists = matching.buffered_iou_distance(r_tracked_stracks, detections_second, 0.5)

        if vis_type == 'full':
            assoc2_dets = [det for det in detections_second] # For detection visualization (2/5)
        if self.logger is not None:
            self.logger.log_dists(dists, mask_match_included=False, which_association=2, frame_no=self.frame_id)

        matches, u_track, u_detection_second, dists_cp = self.conditioned_assignment(dists, MAX_COST_2ND_ASSOC_STEP, r_tracked_stracks, detections_second, prediction_mask, tracklet_mask_dict, img_info)
        if self.logger is not None:
            self.logger.log_dists(dists_cp, mask_match_included=True, which_association=2, frame_no=self.frame_id)
        
        if self.logger is not None:
            strack_pool_ids = [s.track_id for s in r_tracked_stracks]
            self.logger.log_matches(matches, u_track, u_detection_second, strack_pool_ids)
            self.logger.log_det_conf_scores(detections_second)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # The buffered-IoU variant 3/3
        # dists = matching.buffered_iou_distance(unconfirmed, detections, 0.3)
        
        dists = matching.fuse_score(dists, detections)

        # NOTE: At this stage, the unconfirmed tracklets do not have their own masks yet

        if vis_type == 'full':
            assoc3_dets = [det for det in detections]  # For detection visualization (3/5)
        if self.logger is not None:    
            self.logger.log_dists(dists, mask_match_included=False, which_association=3, frame_no=self.frame_id)

        matches, u_unconfirmed, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_UNCONFIRMED_ASSOC_STEP, unconfirmed, detections, prediction_mask, tracklet_mask_dict, img_info)
        if self.logger is not None:
            self.logger.log_dists(dists_cp, mask_match_included=True, which_association=3, frame_no=self.frame_id)
        
        if self.logger is not None:
            strack_pool_ids = [s.track_id for s in unconfirmed]
            self.logger.log_matches(matches, u_unconfirmed, u_detection, strack_pool_ids)
            self.logger.log_det_conf_scores(detections)

        new_confirmed_tracks = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            new_confirmed_tracks.append(unconfirmed[itracked])
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        """ Step 5: Init new stracks"""
        self.feature_db_new_inits = {}

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                if vis_type == 'full':
                    init_track_dets_rej.append(track) # For detection visualization (4/5)
                continue
            track.activate(self.kalman_filter, self.frame_id)

            activated_starcks.append(track)
            if vis_type == 'full':
                init_track_dets_acc.append(track) # For detection visualization (5/5)

    
        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                self._stash_identity_if_ready(track)

           
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # This function, remove_duplicate_stracks(), originally provided with ByteTrack, might cause little bugs 
        # with tracklet management, hence commented out. Kept for reference
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # [re-ID] After all association & state updates...
        # 1) update per-track re-ID features (no masks in decisions)
        self._update_reid_features_for_tracks(frame_img=frame_img)

        # 2) try to reconnect young new tracks to terminated identities
        self.try_reconnect_new_tracks()

        # 3) HArd guarantee: no duplicate export_id among active tracks
        self._ensure_unique_export_ids()

        # 4) finalize export_id for tracks that aged out of reconnection window (confirmed-new)
        self._finalize_old_provisional_export_ids()

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated] # ByteTrack's way
        # output_stracks = [track for track in self.tracked_stracks] # BoT-SORT's way

        if self.logger is not None:
            self.logger.log_local_update_trackets_ids(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
            self.logger.log_state_tracklets_ids(self.tracked_stracks, self.lost_stracks, self.removed_stracks)

        if vis_type == 'full':
            detections_per_assoc_step = {'assoc1': assoc1_dets, 'assoc2': assoc2_dets, 'assoc3': assoc3_dets, 'init_acc': init_track_dets_acc, 'init_rej': init_track_dets_rej}
        else:
            detections_per_assoc_step = None

        removed_tracks_ids = [track.track_id for track in removed_stracks]

    
        return output_stracks, removed_tracks_ids, new_confirmed_tracks, detections_per_assoc_step, all_considered_tracklets_before_correction


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: 1D numpy arrays, assumed L2-normalized.
    """
    if a is None or b is None:
        return None
    return float(np.dot(a, b))

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
