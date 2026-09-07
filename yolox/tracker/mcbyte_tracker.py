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

### Constants ### 
MIN_MASK_AVG_CONF = 0.6
MIN_MM1 = 0.9
MIN_MM2 = 0.05

MAX_COST_1ST_ASSOC_STEP = 0.9
MAX_COST_2ND_ASSOC_STEP = 0.5
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.7


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # NOt sure if this one here is actually required. Well, maybe in case I want to output and/or visualize freshly initialized tracklets in the future...
        self.last_det_tlwh = tlwh ## Extra added

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0 ## Extra added from botsort
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0 ## Extra added from botsort
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
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh ## Extra added

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
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh ## Extra added

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        # ret[2] *= ret[3] # NOTE, Commented out only in the botsort version, not bytetrack version
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    ## Extra added, from botsort
    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    ## Extra added, from botsort
    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    # def to_xyah(self):
    #     return self.tlwh_to_xyah(self.tlwh)

    # This version instead of the one commented out above, according to botsort
    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)


    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
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

   
class McByteTracker(object):
    def __init__(self, args, save_folder, frame_rate=30):
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
        self.gmc = GMC(method=args.cmc_method, verbose=None)

        self.save_folder = save_folder
        log_file_path = osp.join(self.save_folder, "logging_info.txt")
        self.logger = McByteLogger(log_file_path)
 
    
    def conditioned_assignment(self, dists, max_cost, strack_pool, detections, prediction_mask, tracklet_mask_dict,mask_avg_prob_dict, img_info):
        dists_cp = np.copy(dists)

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
                        # It is the case. Update the dists matrix based on the other cue(s)
                        strack = strack_pool[i]
                        det = detections[j]

                        # If there exists a mask for this tracklet
                        strack_id = strack.track_id
                        if strack_id in tracklet_mask_dict.keys():
                            strack_mask_id = tracklet_mask_dict[strack_id]

                            # If this mask is present at the scene right now
                            if strack_mask_id in list(np.unique(prediction_mask))[1:]:

                                # If the mask is confident enough (to be checked only when mask is indeed present at the scene)
                                if mask_avg_prob_dict[strack_mask_id] >= MIN_MASK_AVG_CONF:
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
                                    mask_match_opt_1 = ((prediction_mask[y:ver_bound, x:hor_bound] == strack_mask_id).sum()) / ((prediction_mask == strack_mask_id).sum())
                                    mask_match_opt_2 = ((prediction_mask[y:ver_bound, x:hor_bound] == strack_mask_id).sum()) / ((ver_bound-y) * (hor_bound-x))

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


    def update(self, output_results, img_info, img_size, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, frame_img, vis_type, dets_from_file=False):
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

        # Fix camera motion 
        try:
            warp = self.gmc.apply(frame_img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        except Exception:
            # print("[Frame {}] Internal error while trying to apply the GMC, thus skipping camera motion compensation".format(str(self.frame_id)))
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
        self.logger.log_dists(dists, mask_match_included=False, which_association=1, frame_no=self.frame_id)
                                     
        matches, u_track, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_1ST_ASSOC_STEP, strack_pool, detections, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=1, frame_no=self.frame_id)
        
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
        self.logger.log_dists(dists, mask_match_included=False, which_association=2, frame_no=self.frame_id)

        matches, u_track, u_detection_second, dists_cp = self.conditioned_assignment(dists, MAX_COST_2ND_ASSOC_STEP, r_tracked_stracks, detections_second, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=2, frame_no=self.frame_id)
        
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
        self.logger.log_dists(dists, mask_match_included=False, which_association=3, frame_no=self.frame_id)

        matches, u_unconfirmed, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_UNCONFIRMED_ASSOC_STEP, unconfirmed, detections, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=3, frame_no=self.frame_id)
        
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

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated] # ByteTrack's way
        # output_stracks = [track for track in self.tracked_stracks] # BoT-SORT's way

        self.logger.log_local_update_trackets_ids(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
        self.logger.log_state_tracklets_ids(self.tracked_stracks, self.lost_stracks, self.removed_stracks)

        if vis_type == 'full':
            detections_per_assoc_step = {'assoc1': assoc1_dets, 'assoc2': assoc2_dets, 'assoc3': assoc3_dets, 'init_acc': init_track_dets_acc, 'init_rej': init_track_dets_rej}
        else:
            detections_per_assoc_step = None

        removed_tracks_ids = [track.track_id for track in removed_stracks]

    
        return output_stracks, removed_tracks_ids, new_confirmed_tracks, detections_per_assoc_step, all_considered_tracklets_before_correction


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
