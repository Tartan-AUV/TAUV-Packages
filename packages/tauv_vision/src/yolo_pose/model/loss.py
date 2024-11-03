import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from yolo_pose.model.boxes import box_decode, iou_matrix, box_to_mask
from yolo_pose.model.config import Config


def loss(prediction: (torch.Tensor, ...), truth: (torch.Tensor, ...), config: Config) -> (torch.Tensor, (torch.Tensor, ...)):
    classification, box_encoding, mask_coeff, belief_coeff, affinity_coeff, anchor, mask_prototype, belief_prototypes, affinity_prototypes = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_belief, truth_affinity = truth

    device = classification.device

    n_batch = truth_classification.size(0)

    box = box_decode(box_encoding, anchor)

    iou = iou_matrix(anchor, truth_box)

    match_iou, match_index = torch.max(iou * truth_valid.unsqueeze(1).float(), dim=2)

    # TODO: Handle case where there are no positive matches
    positive_match = match_iou >= config.iou_pos_threshold
    negative_match = match_iou <= config.iou_neg_threshold

    classification_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        match_classification = truth_classification[batch_i, match_index[batch_i]]
        match_classification[~positive_match[batch_i]] = 0

        classification_cross_entropy = F.cross_entropy(
            classification[batch_i],
            match_classification,
            reduction="none"
        )
        n_positive_match = positive_match[batch_i].sum()
        n_selected_negative_match = config.negative_example_ratio * n_positive_match

        background_confidence = F.softmax(classification[batch_i], dim=-1)[:, 0]

        # Maybe change how I'm selecting these negative matches?
        _, selected_negative_match_index = torch.topk(
            torch.where(negative_match[batch_i], -background_confidence, -torch.inf),
            k=n_selected_negative_match
        )
        selected_negative_match_index = selected_negative_match_index.detach()

        selected_match = torch.clone(positive_match[batch_i])
        selected_match[selected_negative_match_index] = True
        selected_match = selected_match.detach()

        classification_loss = (selected_match.float() * classification_cross_entropy).sum()

        classification_losses[batch_i] = classification_loss

    classification_loss = classification_losses.sum() / ((1 + config.negative_example_ratio) * positive_match.sum())

    box_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        # TODO: Check if loss should be over encoding or decoding
        box_loss = F.smooth_l1_loss(
            box[batch_i, positive_match[batch_i]],
            truth_box[batch_i, match_index[batch_i, positive_match[batch_i]]],
            reduction="none"
        ).sum()

        box_losses[batch_i] = box_loss

    box_loss = box_losses.sum() / positive_match.sum()

    mask_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        mask_loss = torch.tensor(0, dtype=torch.float, device=device)

        for match_i in positive_match[batch_i].nonzero():
            match_i = int(match_i)
            match_mask = torch.sum(mask_coeff[batch_i, match_i].unsqueeze(1).unsqueeze(2) * mask_prototype[batch_i], dim=0)
            match_mask = F.sigmoid(match_mask)
            match_mask = torch.clamp(match_mask, min=1e-4)

            with torch.no_grad():
                truth_match_mask = (truth_seg_map[batch_i] == truth_classification[batch_i, match_index[batch_i, match_i]]).float()
                truth_match_mask_resized = F.interpolate(
                    truth_match_mask.unsqueeze(0).unsqueeze(0),
                    match_mask.size(),
                    mode="bilinear",
                ).squeeze(0).squeeze(0)

            if truth_match_mask_resized.sum() == 0:
                continue

            mask_cross_entropy = F.binary_cross_entropy(
                match_mask.reshape(-1),
                truth_match_mask_resized.reshape(-1),
                reduction="none",
            )

            box_mask = box_to_mask(
                truth_box[batch_i, match_index[batch_i, match_i]],
                match_mask.size()
            )

            mask_loss += (box_mask.reshape(-1) * mask_cross_entropy).sum() / truth_match_mask_resized.sum()

        mask_losses[batch_i] = mask_loss

    mask_loss = mask_losses.sum() / positive_match.sum()

    # Now predict coeffs for belief and affinity
    # Assemble in the same way masks are assembled
    # Take loss for each truth detection and average

    belief_losses = torch.zeros(n_batch, device=device)
    affinity_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        belief_loss = torch.tensor(0, dtype=torch.float, device=device)
        affinity_loss = torch.tensor(0, dtype=torch.float, device=device)

        for belief_prototype, affinity_prototype in zip(belief_prototypes, affinity_prototypes):
            for match_i in positive_match[batch_i].nonzero():
                match_i = int(match_i)

                coeffs = belief_coeff[batch_i, match_i]
                # coeffs = coeffs / torch.sum(torch.abs(coeffs))
                match_belief = torch.matmul(
                    coeffs,
                    belief_prototype[batch_i].reshape(belief_prototype.size(1), -1)
                ).reshape(belief_coeff.size(2), belief_prototype.size(2), belief_prototype.size(3))
                match_belief = torch.clamp(F.sigmoid(match_belief), min=1e-4, max=1-1e-4)
                # max_belief, _ = torch.max(match_belief.reshape(match_belief.size(0), -1), dim=1)
                # match_belief /= max_belief.unsqueeze(1).unsqueeze(2)
                match_affinity = torch.matmul(
                    affinity_coeff[batch_i, match_i],
                    affinity_prototype[batch_i].reshape(affinity_prototype.size(1), -1)
                ).reshape(affinity_coeff.size(2), affinity_prototype.size(2), affinity_prototype.size(3))
                match_affinity = 2 * (torch.clamp(F.sigmoid(match_affinity), min=1e-4) - 0.5)

                truth_match_belief = truth_belief[batch_i, match_index[batch_i, match_i]]
                truth_match_affinity = truth_affinity[batch_i, match_index[batch_i, match_i]]

                truth_match_belief_resized = F.interpolate(
                    truth_match_belief.unsqueeze(0),
                    match_belief.size()[1:3],
                    mode="bilinear",
                ).squeeze(0)

                truth_match_affinity_resized = F.interpolate(
                    truth_match_affinity.unsqueeze(0),
                    match_affinity.size()[1:3],
                    mode="bilinear",
                ).squeeze(0)

                beta = 1 - truth_match_belief_resized.mean()
                # beta = 0.5
                belief_loss_map = -beta * truth_match_belief_resized * torch.log(match_belief) - (1 - beta) * (1 - truth_match_belief_resized) * torch.log(1 - match_belief)

                affinity_loss_map = F.mse_loss(
                    match_affinity,
                    truth_match_affinity_resized,
                    reduction="none",
                )

                belief_loss += belief_loss_map.mean()
                affinity_loss += affinity_loss_map.mean()

        belief_losses[batch_i] = belief_loss
        affinity_losses[batch_i] = affinity_loss

    # fig, axs = plt.subplots(4)
    # axs[0].imshow(truth_match_belief_resized[0].detach().cpu())
    # im = axs[1].imshow(match_belief[0].detach().cpu())
    # axs[2].imshow(belief_loss_map[0].detach().cpu())
    # axs[3].imshow(coeffs.detach().cpu())
    # fig.colorbar(im)

    # fig, axs = plt.subplots(3)
    # axs[0].imshow(truth_match_affinity_resized[0].detach().cpu())
    # im = axs[1].imshow(match_affinity[0].detach().cpu())
    # axs[2].imshow(affinity_loss_map[0].detach().cpu())
    # fig.colorbar(im)
    # plt.show()

    belief_loss = belief_losses.sum() / positive_match.sum()
    affinity_loss = affinity_losses.sum() / positive_match.sum()

    total_loss = classification_loss + box_loss + mask_loss + belief_loss + affinity_loss
    # total_loss = belief_loss
    # total_loss = 100 * belief_loss

    return total_loss, (classification_loss, box_loss, mask_loss, belief_loss, affinity_loss)
