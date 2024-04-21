from operator import gt
from turtle import back
import torch
import random

from ..builder import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class HieAssigner(BaseAssigner):

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'), 
                 assign_metric='kl',
                 topk=[2, 1],
                 ratio=1,
                 inside=False):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.ratio = ratio
        self.inside = inside

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):

        # import IPython
        # IPython.embed()
        # exit()
        bboxes = bboxes.priors
        gt_labels = gt_bboxes.labels
        gt_bboxes = gt_bboxes.bboxes
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        '''
        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)
        bboxes2 = self.anchor_rescale(bboxes, self.ratio)
        overlaps2 = self.iou_calculator(gt_bboxes, bboxes2, mode=self.assign_metric)
        '''

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)
        bboxes2 = self.anchor_rescale(bboxes, self.ratio)
        overlaps2 = self.iou_calculator(gt_bboxes, bboxes2, mode=self.assign_metric)



        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        k1 = self.topk[0]
        k2 = self.topk[1] 
        #k3 = self.topk[2]

        ## two_step anchor assigning
        assigned_gt_inds = self.assign_wrt_ranking(overlaps, k1, gt_labels)
        #assign_result = self.reassign_wrt_ranking_v5(assigned_gt_inds, overlaps2, k2, gt_labels)
        assign_result = self.reassign_wrt_ranking(assigned_gt_inds, overlaps2, k2, gt_labels)

        ## filter out low quality candidates
        if self.inside==True:
            num_anchors = bboxes.size(0)
            num_gts = gt_bboxes.size(0)

            anchor_cx = (bboxes[...,0]+bboxes[...,2])/2
            anchor_cy = (bboxes[...,1]+bboxes[...,3])/2
            ext_gt_bboxes = gt_bboxes[:,None,:].expand(num_gts, num_anchors, 4)
            left = anchor_cx - ext_gt_bboxes[...,0]
            right = ext_gt_bboxes[..., 2] - anchor_cx
            top = anchor_cy - ext_gt_bboxes[..., 1]
            bottom = ext_gt_bboxes[..., 3] - anchor_cy

            bbox_targets = torch.stack((left, top, right, bottom), -1)
            inside_flag = bbox_targets.min(-1)[0] > 0
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[(assign_result.gt_inds-1).clamp(min=0), length]
            assign_result.gt_inds *= inside_mask

    
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self,  overlaps, k, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return assigned_gt_inds

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(k, dim=1, largest=True, sorted=True)


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0
        #assign wrt ranking
        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return assigned_gt_inds



    def reassign_wrt_ranking(self, assign_result, overlaps, k, gt_labels=None):

        
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        mask1 = assign_result <= 0  
        mask2 = assign_result > 0 

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(k, dim=1, largest=True, sorted=True)

        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        #assign wrt ranking
        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        assigned_gt_inds = assigned_gt_inds * mask1 + assign_result * mask2


        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    


    def anchor_rescale(self, bboxes, ratio):
        center_x2 = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y2 = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w2 = bboxes[..., 2] - bboxes[..., 0]
        h2 = bboxes[..., 3] - bboxes[..., 1]
        bboxes[..., 0] = center_x2 - w2*ratio/2
        bboxes[..., 1] = center_y2 - h2*ratio/2
        bboxes[..., 2] = center_x2 + w2*ratio/2
        bboxes[..., 3] = center_y2 + h2*ratio/2
        
        return bboxes


    def anchor_offset(self, bboxes, ratio):
        center_x2 = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y2 = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w2 = bboxes[..., 2] - bboxes[..., 0]
        h2 = bboxes[..., 3] - bboxes[..., 1]
        offset_x = w2 * (1 - ratio)
        offset_y = h2 * (1 - ratio)
        center_x3 = center_x2 + offset_x
        center_y3 = center_y2 + offset_y
        bboxes[..., 0] = center_x3 - w2/2
        bboxes[..., 1] = center_y3 - h2/2
        bboxes[..., 2] = center_x3 + w2/2
        bboxes[..., 3] = center_y3 + h2/2

        return bboxes

    def anchor_reshape(self, bboxes, ratio):
        center_x2 = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y2 = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w2 = bboxes[..., 2] - bboxes[..., 0]
        h2 = bboxes[..., 3] - bboxes[..., 1]
        aspect_ratio = w2/h2
        scale = w2 * h2
        new_asratio = ratio * aspect_ratio 
        new_w2 = torch.sqrt(scale* new_asratio)
        new_h2 = torch.sqrt(scale/new_asratio)
        bboxes[..., 0] = center_x2 - new_w2/2
        bboxes[..., 1] = center_y2 - new_h2/2
        bboxes[..., 2] = center_x2 + new_w2/2
        bboxes[..., 3] = center_y2 + new_h2/2

        return bboxes







 

    


        
