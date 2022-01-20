import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask):
    heatmap_loss_log = []
    paf_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask.unsqueeze(1).repeat([1, pafs_t.shape[1], 1, 1])
    heatmap_masks = ignore_mask.unsqueeze(1).repeat(
        [1, heatmaps_t.shape[1], 1, 1])

    # compute loss on each stage
    for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
        stage_pafs_t = pafs_t.clone()
        stage_heatmaps_t = heatmaps_t.clone()
        stage_paf_masks = paf_masks.clone()
        stage_heatmap_masks = heatmap_masks.clone()

        if pafs_y.shape != stage_pafs_t.shape:
            with torch.no_grad():
                stage_pafs_t = F.interpolate(
                    stage_pafs_t, pafs_y.shape[2:], mode='bilinear',
                    align_corners=True)
                stage_heatmaps_t = F.interpolate(
                    stage_heatmaps_t, heatmaps_y.shape[2:],
                    mode='bilinear', align_corners=True)
                stage_paf_masks = F.interpolate(
                    stage_paf_masks, pafs_y.shape[2:]) > 0
                stage_heatmap_masks = F.interpolate(
                    stage_heatmap_masks, heatmaps_y.shape[2:]) > 0

        with torch.no_grad():
            stage_pafs_t[stage_paf_masks == 1] = pafs_y.detach()[
                stage_paf_masks == 1]
            stage_heatmaps_t[stage_heatmap_masks == 1] = heatmaps_y.detach()[
                stage_heatmap_masks == 1]

        pafs_loss = mean_square_error(pafs_y, stage_pafs_t)
        heatmaps_loss = mean_square_error(heatmaps_y, stage_heatmaps_t)

        total_loss += pafs_loss + heatmaps_loss

        paf_loss_log.append(pafs_loss)
        heatmap_loss_log.append(heatmaps_loss)

    return total_loss, paf_loss_log, heatmap_loss_log


def mean_square_error(pred, target):
    assert pred.shape == target.shape, 'x and y should in same shape'
    return torch.sum((pred - target) ** 2) / target.nelement()


class Custom_Loss(nn.Module):
    def __init__(self):
        super(Custom_Loss, self).__init__()

    def forward(self, pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask):

        return compute_loss(
            pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask)


# class Custom_Loss(nn.Module):
#     def __init__(self):
#         super(Custom_Loss, self).__init__()
#         self.mse_loss = nn.MSELoss(reduction='mean')
#
#     def forward(self, out_list):
#         loss_dict = out_list[-1]
#         out_dict = dict()
#         weight_dict = dict()
#         for key, item in loss_dict.items():
#             out_dict[key] = self.mse_loss(*item['params'])
#             weight_dict[key] = item['weight'].mean().item()
#
#         loss = 0.0
#         for key in out_dict:
#             loss += out_dict[key] * weight_dict[key]
#
#         out_dict['loss'] = loss
#         return out_dict
