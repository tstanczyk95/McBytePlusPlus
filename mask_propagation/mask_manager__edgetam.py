OVERLAP_MEASURE_VARIANT = 1
OVERLAP_VARIANT_2_GRID_STEP = 10
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.1

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./mask_propagation/EdgeTAM/checkpoints/edgetam.pt"
model_cfg  = "configs/edgetam.yaml"

# You can also run it with SAM2.1 instead. It will be more accurate, but slower
# checkpoint = "./mask_propagation/EdgeTAM/checkpoints/sam2.1_hiera_large.pt"
# model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"

MM1_MASK_UPDATE_THRESHOLD = 0.5

class MaskManager(object):
    def __init__(self, input_frame_dir, args):
        self.with_mask_removal = args.with_mask_removal
        self.with_mask_update = args.with_mask_update
        self.mask_update_every_k_frames = args.mask_update_every_k_frames

        # print("self.with_mask_removal", self.with_mask_removal)
        # print("self.with_mask_update", self.with_mask_update)
        # print("self.mask_update_every_k_frames", self.mask_update_every_k_frames)

        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0
        self.SAM_START_FRAME = 1

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.state = self.predictor.init_state(video_path=input_frame_dir)

        self.handled_prompt_frames = set()
        self.current_iter = None

        self.last_frame_out_obj_ids = None
        self.last_frame_out_mask_logits = None


    def get_updated_masks(self, frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids):
        out_obj_ids = None
        out_mask_logits = None
            
        if frame_id == self.SAM_START_FRAME + 1 + self.init_delay_counter and online_tlwhs is not None:
            out_obj_ids, out_mask_logits = self.initialize_first_masks(frame_id, online_tlwhs, online_ids)
        elif frame_id > self.SAM_START_FRAME + 1 + self.init_delay_counter:
            self.update_masks(frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, out_mask_logits = next(self.current_iter)

        self.last_frame_out_obj_ids = out_obj_ids
        self.last_frame_out_mask_logits = out_mask_logits

        if out_obj_ids is None:
            prediction = None
        else:
            prediction = {"mask_ids" : out_obj_ids, "mask_logits" : out_mask_logits}

        return prediction, self.tracklet_mask_dict.copy()


    def initialize_first_masks(self, frame_id, online_tlwhs, online_ids):
        image_boxes_list = []
        new_tracks_id = []

        # Based on McByte/ByteTrack mechanisms, the tracklets created at the first frame 
        # will already be considered fully activated, thus as the tracked tracklets 
        # (stracks). Therefore using the online_tlwhs coming from the tracker output.
        for i, ot in enumerate(online_tlwhs):

            ### Avoiding creation of the "weird" masks (from overlapped/occluded subjects) ###
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(ot, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(ot, track_BBs_with_lower_bottom)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue
            ### </> ###

            image_boxes_list.append(np.array([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]], dtype=np.float32))
            new_tracks_id.append(online_ids[i])

        if len(image_boxes_list) == 0:
            # Delay the whole process
            self.init_delay_counter += 1
            return None 
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for tr_id, im_box in zip(new_tracks_id, image_boxes_list): 
                    # Just add them. Their values will be needed only from the next 
                    # frame (after first propagation)
                    # frame_id-2 : -2, because -1 for the previous frame and another -1, 
                    # because EdgeTam counts from 0 while McByte counts from 1
                    self.predictor.add_new_points_or_box(
                        inference_state=self.state, frame_idx=frame_id-2, obj_id=tr_id, box=im_box
                    )

                self.current_iter = self.predictor.propagate_in_video(self.state)
                try:
                    # Next frame outputs (propagation by one frame after adding the first masks)
                    _, out_obj_ids, out_mask_logits = next(self.current_iter) # frame 1 (EdgeTAM: frame 0)
                    _, out_obj_ids, out_mask_logits = next(self.current_iter) # frame 2 (EdgeTAM: frame 1)
                except StopIteration:
                    print("[Adding first masks] No more frames after the addition!")

            self.tracklet_mask_dict = dict(zip(new_tracks_id, new_tracks_id))
            self.mask_color_dict = dict(zip(new_tracks_id, new_tracks_id))

            return out_obj_ids, out_mask_logits


    def reseed_at_frame(self, frame_id, existing_masks_dict, image_boxes_list, new_tracks_id):
        """
        existing_masks_dict: {obj_id: (H,W) boolean/float mask at frame_idx}
        new_prompts: list of dicts as returned by new_prompts_at_frame
        """
        self.tracklet_mask_dict = {}
        self.mask_color_dict = {}

        self.predictor.reset_state(self.state)
        # re-add existing objects using their exact masks at this frame
        for oid, m in existing_masks_dict.items():
            m = m.astype(np.float32)  # EdgeTAM expects float mask
            self.predictor.add_new_mask(inference_state=self.state, frame_idx=frame_id, obj_id=oid, mask=m)
            self.tracklet_mask_dict[oid] = oid

        # add all new objects on the same frame
        for tr_id, im_box in zip(new_tracks_id, image_boxes_list):
            self.predictor.add_new_points_or_box(
                inference_state=self.state, frame_idx=frame_id, obj_id=tr_id, box=im_box
            )
            self.tracklet_mask_dict[tr_id] = tr_id

        self.mask_color_dict = self.tracklet_mask_dict.copy()

    # --- Mask addition ---
    # Use the last_frame data and start propagating from that frame. Then do one more propagation on the current frame.
    # E.g. we are adding new objects as "backlog" to frame 50 (which has already been propagated). Add these new objects at 
    # frame 50, together with the existing objects saved in self.last_frame_out_obj_ids and self.last_frame_out_mask_logits.
    # Then perform two more propagations: for frame 50 and frame 51.
    #
    # --- OPTIONAL mask removal and refreshment ---
    # * [Every single frame? Every 25 frames?] Compute mm1 for each mask-tracklet (iterate over masks)
    # * if mm1 low (threshold to be defined, e.g. 0.5?) for the given mask, then:
    #       - mark this mask to be removed
    #       - add the related tracklet id to self.awaiting_mask_tracklet_ids
    def update_masks(self, frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids):     
        renew_mask_trackled_ids = []

        if not self.with_mask_removal:
            removed_tracks_ids = []

        if self.with_mask_update:

            # McByte counts frames from 1
            # if frame_id % 1 == 0: # % 25 % 1
            if frame_id % self.mask_update_every_k_frames == 0:            
                
                for i in range(len(self.last_frame_out_obj_ids)):
                    # NOTE: specific to EdgeTAM/SAM2 mask_manager version only! (NOT for Cutie)
                    track_id = self.last_frame_out_obj_ids[i]

                    # Only perform this for active (not lost) tracklets
                    if track_id not in online_ids:
                        continue

                    mask_all_valid_pixels = (self.last_frame_out_mask_logits[i][0] > 0.0).sum()
                    if mask_all_valid_pixels == 0:
                        renew_mask_trackled_ids.append(track_id)
                        self.awaiting_mask_tracklet_ids.append(track_id)
                        continue

                    ### Compute mm1 ("mc" in the paper) ###
                    img_h, img_w = self.last_frame_out_mask_logits[0][0].shape
                                        
                    # Get the tracklet coordinates and prepare them for computing mm1
                    track_list_position = online_ids.index(track_id)
                    track_tlwh = online_tlwhs[track_list_position]
                    x, y, w, h = track_tlwh

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

                    mask_pixels_in_bb = (self.last_frame_out_mask_logits[i][0][y:ver_bound, x:hor_bound] > 0.0).sum()
                    mask_match_opt_1 = mask_pixels_in_bb / mask_all_valid_pixels

                    if mask_match_opt_1 < MM1_MASK_UPDATE_THRESHOLD:
                        renew_mask_trackled_ids.append(track_id)
                        self.awaiting_mask_tracklet_ids.append(track_id)

    
        if len(new_tracks) > 0 or len(self.awaiting_mask_tracklet_ids)> 0:
            image_boxes_list = []
            new_tracks_id = []

            ### Try to create masks for the tracklets awaiting from the previous frames. Avoiding creation of the "weird" masks (from overlapped/occluded subjects) ###
            for i, amti in enumerate(self.awaiting_mask_tracklet_ids):
                # Take into account only these tracklets with ids from awaiting_mask_tracklet_ids, that are actually active, not marked as lost. Thus 
                # only these tracklets, that have been returned by the McByte's update function (their IDs in online_ids and coords in online_tlwhs)
                if not amti in online_ids:
                    continue

                amt_index = online_ids.index(amti)
                amt_tlwh = online_tlwhs[amt_index]

                track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(amt_tlwh, online_tlwhs)
                overlap = get_overlap_with_lower_bottom_tracklets(amt_tlwh, track_BBs_with_lower_bottom)

                # If the overlap is not too big, initiate mask creation for this tracklet and set it to be removed from the awaiting list (awaiting_mask_tracklet_ids)
                if overlap < MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                    image_boxes_list.append(np.array([amt_tlwh[0], amt_tlwh[1], amt_tlwh[0] + amt_tlwh[2], amt_tlwh[1] + amt_tlwh[3]], dtype=np.float32))
                    new_tracks_id.append(amti)

            # Remove from the awaiting list (awaiting_mask_tracklet_ids) the tracklets which are going to have their masks created
            for nti in new_tracks_id:
                self.awaiting_mask_tracklet_ids.remove(nti)
            ### </> ###

            ### Avoiding creation of the "weird" masks (from overlapped subjects) - for the new tracklets from the current frame ###
            for i, nt in enumerate(new_tracks):
                track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(nt.last_det_tlwh, online_tlwhs)
                
                # For the considered tracklet nt, measure the overlap with the tracklets retrieved as above (2 variants possible)
                overlap = get_overlap_with_lower_bottom_tracklets(nt.last_det_tlwh, track_BBs_with_lower_bottom)

                # If the overlap exceeds the defined threshold, then do not create the mask for this tracklet
                # Keep the tracklet ID in a separate list, awaiting_mask_tracklet_ids
                if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                    self.awaiting_mask_tracklet_ids.append(nt.track_id)
                    continue

                image_boxes_list.append(np.array([nt.last_det_tlwh[0], nt.last_det_tlwh[1], nt.last_det_tlwh[0] + nt.last_det_tlwh[2], nt.last_det_tlwh[1] + nt.last_det_tlwh[3]], dtype=np.float32))                        
                new_tracks_id.append(nt.track_id)
            ### </> ###


            if len(image_boxes_list) > 0 or len(removed_tracks_ids) > 0 or len(renew_mask_trackled_ids) > 0:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

                    # We need the masks for all existing objects at this frame to reseed.
                    existing_masks_at_f = {}
                    for i, oid in enumerate(self.last_frame_out_obj_ids):
                        if oid in removed_tracks_ids: # Assumption: tracklet_id == mask_id
                            # print("--- TRACKLET ID={} - MASK (TO BE) REMOVED ---".format(oid))
                            continue
                        if oid in renew_mask_trackled_ids:
                            # print("--- TRACKLET ID={} - MASK SET TO BE CREATED A NEW ---".format(oid))
                            continue
                        m = (self.last_frame_out_mask_logits[i] > 0.0).detach().cpu().numpy()
                        if m.ndim == 3:
                            m = m[0]
                        existing_masks_at_f[oid] = m

                    # frame_id-2 : -2, because -1 for the previous frame and another -1, because EdgeTam counts from 0 and McByte counts from 1
                    # Do not modify it or else you'll get trashy/non-matching outputs
                    self.reseed_at_frame(frame_id-2, existing_masks_at_f, image_boxes_list, new_tracks_id)

                    self.current_iter = self.predictor.propagate_in_video(self.state)

                    # Do not modify/do not remove it or else you'll get trashy/non-matching outputs
                    next(self.current_iter)

    

def get_tracklets_with_lower_bottom(new_tracklet_tlwh, online_tlwhs):
    nt_y = new_tracklet_tlwh[1]
    nt_h = new_tracklet_tlwh[3]

    # Find all the tracklets, both the new ones (new_tracks) and the exisiting ones (online_tlwhs), 
    # with their bottom coordinate higher (thus being at the lower position) than the bottom coordinate of the considered tracklet nt
    track_BBs_with_lower_bottom = []
    nt_bottom = nt_y + nt_h

    for ot in online_tlwhs:
        if ot[1] + ot[3] > nt_bottom: # >, not >= so as to avoid the conflict with itself (also included in online_tlwhs)
            track_BBs_with_lower_bottom.append(ot)

    return track_BBs_with_lower_bottom


def get_overlap_with_lower_bottom_tracklets(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    overlap = 0
    
    if OVERLAP_MEASURE_VARIANT == 1:
        overlap = get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom)
    elif OVERLAP_MEASURE_VARIANT == 2:
        overlap = get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom)

    return overlap


# Variant 1: Find the maximal overlap (intersection) of two tracklets, between the new one and exisiting ones from the list. 
# Then divide it by the size of the considered tracklet bounding box
def get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = new_tracklet_tlwh[0]
    nt_y = new_tracklet_tlwh[1]
    nt_w = new_tracklet_tlwh[2]
    nt_h = new_tracklet_tlwh[3]
    
    max_overlap_part = 0 # between 0 and 1

    for lb in track_BBs_with_lower_bottom:
        x_dist = min(nt_x+nt_w, lb[0]+lb[2]) - max(nt_x, lb[0])
        y_dist = min(nt_y+nt_h, lb[1]+lb[3]) - max(nt_y, lb[1])

        if x_dist < 0 or y_dist < 0:
            overlap_area = 0
        else:
            overlap_area = x_dist * y_dist

        overlap_part = overlap_area / (nt_w * nt_h)
        if max_overlap_part < overlap_part:
            max_overlap_part = overlap_part

            if max_overlap_part == 1:
                break

    return max_overlap_part


# Variant 2: Check each pixel (or every 10th pixel vertically and horizontally - imagine a grid) from the considered
# tracklet bounding box if it is also within the bounding box of another tracklet (from all the ones retrieved) - based on their coordinates.
# The overlap measure is then the ratio of the number of occupied pixels to the size of the boudnind box (in terms of each 10 pixel vert. and hor.)
def get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = int(new_tracklet_tlwh[0])
    nt_y = int(new_tracklet_tlwh[1])
    nt_w = int(new_tracklet_tlwh[2])
    nt_h = int(new_tracklet_tlwh[3])

    point_overlap_counter = 0

    for grid_row in range(nt_y, nt_y+nt_h, OVERLAP_VARIANT_2_GRID_STEP):
        for grid_col in range(nt_x, nt_x+nt_w, OVERLAP_VARIANT_2_GRID_STEP):
            for lb in track_BBs_with_lower_bottom:
                if lb[0] <= grid_col and grid_col <= lb[0] + lb[2] and lb[1] <= grid_row and grid_row <= lb[1] + lb[3]:
                    point_overlap_counter += 1
                    break

    overlap_part = point_overlap_counter / (len(range(nt_y, nt_y+nt_h, OVERLAP_VARIANT_2_GRID_STEP)) * len(range(nt_x, nt_x+nt_w, OVERLAP_VARIANT_2_GRID_STEP)))
    
    return overlap_part


def update_tracklet_mask_dict_after_mask_addition(tracklet_mask_dict, mask_color_dict, added_tracklet_ids, added_mask_ids, mask_color_counter): 
    # Part 1/2: Update the mask ids in tracklet_mask_dict
    for k,v in zip(added_tracklet_ids, added_mask_ids):
        tracklet_mask_dict[k] = v

    # Part 2/2: Update the mask ids in mask_color_dict
    for mi in added_mask_ids:
        mask_color_counter += 1
        mask_color_dict[mi] = mask_color_counter

    return mask_color_counter


def update_tracklet_mask_dict_after_mask_removal(tracklet_mask_dict, mask_color_dict, removed_mask_ids):
    # Part 1/2: Update the mask ids in tracklet_mask_dict
    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in tracklet_mask_dict.keys():
        if tracklet_mask_dict[k] in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids:
                if tracklet_mask_dict[k] > rmi:
                    if not k in decrement_mask_id_dict.keys():
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del tracklet_mask_dict[entry]

    for k in decrement_mask_id_dict.keys():
        tracklet_mask_dict[k] -= decrement_mask_id_dict[k]


    # Part 2/2: Update the mask ids in mask_color_dict

    # Saved and returned in case the last element (with the highest number) was deleted
    mask_color_counter = max(list(mask_color_dict.values()), default=0)

    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in mask_color_dict.keys():
        if k in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids: 
                if k > rmi:
                    if not k in decrement_mask_id_dict.keys():
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del mask_color_dict[entry]

    mask_color_keys = list(decrement_mask_id_dict.keys())
    for mc in mask_color_keys:
        new_key = mc - decrement_mask_id_dict[mc]
        mask_color_dict[new_key] = mask_color_dict[mc]
        del mask_color_dict[mc]

    return mask_color_counter