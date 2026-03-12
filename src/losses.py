import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def mcc_loss(sk_feature, photo_feature):
    l1 = nn.L1Loss()
    sk_feature = F.normalize(sk_feature)
    photo_feature = F.normalize(photo_feature)
    
    sk2sk_sim = sk_feature @ sk_feature.t()
    ph2ph_sim = photo_feature @ photo_feature.t()
    
    mcc_sk = torch.tensor(0.1)
    mcc_ph = torch.tensor(0)
    
    loss_mcc_sk = l1(sk2sk_sim.mean(), mcc_sk.to(device)) * 4
    loss_mcc_ph = l1(ph2ph_sim.mean(),mcc_ph.to(device)) * 8
    
    return loss_mcc_sk + loss_mcc_ph

def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)


def _directional_batch_hard_triplet(distance_matrix, category_labels, instance_labels, margin):
    positive_distance = distance_matrix.diagonal()

    same_category = category_labels.unsqueeze(1).eq(category_labels.unsqueeze(0))
    different_instance = ~instance_labels.unsqueeze(1).eq(instance_labels.unsqueeze(0))

    hard_negative_mask = same_category & different_instance
    fallback_negative_mask = different_instance

    hard_negative_distance = distance_matrix.masked_fill(~hard_negative_mask, float('inf')).min(dim=1).values
    fallback_negative_distance = distance_matrix.masked_fill(~fallback_negative_mask, float('inf')).min(dim=1).values
    hard_negative_distance = torch.where(
        torch.isinf(hard_negative_distance),
        fallback_negative_distance,
        hard_negative_distance,
    )

    valid_rows = ~torch.isinf(hard_negative_distance)
    if not valid_rows.any():
        return distance_matrix.new_tensor(0.0)

    triplet_loss = F.relu(
        positive_distance[valid_rows] - hard_negative_distance[valid_rows] + margin
    )
    return triplet_loss.mean()


def batch_hard_triplet_loss(sk_feature, photo_feature, category_labels, instance_labels, margin=0.3):
    sk_feature = F.normalize(sk_feature, dim=1)
    photo_feature = F.normalize(photo_feature, dim=1)

    distance_matrix = 1.0 - sk_feature @ photo_feature.t()
    loss_sk_to_photo = _directional_batch_hard_triplet(
        distance_matrix, category_labels, instance_labels, margin
    )
    loss_photo_to_sk = _directional_batch_hard_triplet(
        distance_matrix.t(), category_labels, instance_labels, margin
    )

    return 0.5 * (loss_sk_to_photo + loss_photo_to_sk)


def loss_fn(args, model, features, mode='train'):
    photo_features_norm, sk_feature_norm, photo_aug_tensor, sk_aug_tensor, \
        category_labels, instance_labels, pos_logits, sk_logits, photo_features, sk_features = features

    category_labels = category_labels.to(pos_logits.device)
    instance_labels = instance_labels.to(pos_logits.device)
    loss_mcc = mcc_loss(photo_features, sk_features)
    loss_ce_photo = F.cross_entropy(pos_logits, category_labels)
    loss_ce_sk = F.cross_entropy(sk_logits, category_labels)
    
    with torch.no_grad():
        photo_aug_features = model.model_distill.encode_image(photo_aug_tensor)
        sk_aug_features = model.model_distill.encode_image(sk_aug_tensor)
    loss_distill_photo = cross_loss(photo_features, photo_aug_features, args)
    loss_distill_sk = cross_loss(sk_features, sk_aug_features, args)
    
    # photo_aug_features = (photo_aug_features / photo_aug_features.norm(dim=-1, keepdim=True))
    # sk_aug_features = (sk_aug_features / sk_aug_features.norm(dim=-1, keepdim=True))
    
    margin = getattr(args, "triplet_margin", 0.3)
    loss_triplet = batch_hard_triplet_loss(
        sk_feature_norm, photo_features_norm, category_labels, instance_labels, margin=margin
    )
    loss_photo_skt = cross_loss(photo_features, sk_features, args)
    
    loss_distill = loss_distill_photo + loss_distill_sk

    w_triplet = getattr(args, "w_triplet", getattr(args, "alpha", 1.0))
    w_photo_skt = getattr(args, "w_photo_skt", getattr(args, "beta", 1.0))
    w_distill = getattr(args, "w_distill", getattr(args, "gamma", 1.0))
    w_ce = getattr(args, "w_ce", 1.0)
    w_mcc = getattr(args, "w_mcc", getattr(args, "lambd", 1.0))

    return (
        w_triplet * loss_triplet
        + w_photo_skt * loss_photo_skt
        + w_distill * loss_distill
        + w_ce * (loss_ce_photo + loss_ce_sk)
        + w_mcc * loss_mcc
    )
