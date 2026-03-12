from collections import defaultdict
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from src.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.utils import load_clip_to_cpu, get_all_categories
from src.losses import loss_fn
    
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class JigsawHead(nn.Module):
    def __init__(self, feature_dim=512, num_permutations=24, num_heads=8, num_layers=2):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, 2, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_permutations),
        )
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill
    ):
        super().__init__()
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, clip_model, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, clip_model, type='sketch')
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        
        self.model_distill = clip_model_distill
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1
        self.jigsaw_head = JigsawHead(
            feature_dim=512,
            num_permutations=math.factorial(getattr(cfg, 'jigsaw_grid', 2) ** 2),
        )
    
    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
        else:
            prompt_learner = self.prompt_learner_sketch
        # tokenized_prompts = prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            tokenized_prompts,
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = prompt_learner(classnames)
        
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        ) # (n_classes, 512)
        
        image_features = self.image_encoder(
                img_tensor.type(self.dtype), shared_ctx, deep_compound_prompts_vision
            ) # (batch_size, 768)
            
        x_a = self.adapter_photo(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # image_features = F.normalize(image_features, dim=-1)
        # text_features = F.normalize(text_features, dim=-1)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize, image_features
        
    def forward(self, x, classnames):
        (
            photo_tensor,
            sk_tensor,
            photo_aug_tensor,
            sk_aug_tensor,
            neg_tensor,
            sk_jigsaw_tensor,
            perm_label,
            category_label,
        ) = x
        pos_logits, photo_features_norm, photo_feature = self.get_logits(photo_tensor, classnames)
        sk_logits, sk_feature_norm, sk_feature = self.get_logits(sk_tensor, classnames, type='sketch')
        _, neg_feature_norm, neg_feature = self.get_logits(neg_tensor, classnames)
        _, sk_jigsaw_feature_norm, sk_jigsaw_feature = self.get_logits(sk_jigsaw_tensor, classnames, type='sketch')
        
        return photo_features_norm, sk_feature_norm, photo_aug_tensor, sk_aug_tensor, \
            neg_feature_norm, category_label, perm_label, pos_logits, sk_logits, \
            photo_feature, sk_feature, neg_feature, sk_jigsaw_feature
            
    def extract_feature(self, image, classname, type='photo'):
        _, feature, _ = self.get_logits(image, classnames=classname, type=type)
        return feature

    def compute_jigsaw_logits(self, first_feature, second_feature):
        tokens = torch.stack(
            [F.normalize(first_feature, dim=-1), F.normalize(second_feature, dim=-1)],
            dim=1,
        )
        tokens = tokens.to(self.jigsaw_head.positional_embedding.dtype)
        tokens = tokens + self.jigsaw_head.positional_embedding
        encoded = self.jigsaw_head.encoder(tokens)
        logits = self.jigsaw_head.classifier(encoded.reshape(encoded.shape[0], -1))
        return logits
            
class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = load_clip_to_cpu(args)
        
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(args, design_details=design_details)
        
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.best_metric = 0.0
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model, clip_model_distill=clip_model_distill)
    
        self.val = defaultdict(
            lambda: {
                "val_sk_features": [],
                "val_sk_names": [],
                "val_img_features": [],
                "val_img_names": [],
            }
        )
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data, classname):
        return self.model(data, classname)
    
    def training_step(self, batch, batch_idx):
        classname = get_all_categories(self.args)
        features = self.forward(batch, classname)
        
        loss = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        classnames = get_all_categories(self.args, mode="test")
        image_tensor, category_idx, sample_names = batch
        category_indices = category_idx.detach().cpu().tolist()
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='sketch')
            for idx, category in enumerate(category_indices):
                bucket = self.val[category]
                bucket["val_sk_features"].append(feat[idx].detach().cpu())
                bucket["val_sk_names"].append(sample_names[idx])
        else:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='photo')
            for idx, category in enumerate(category_indices):
                bucket = self.val[category]
                bucket["val_img_features"].append(feat[idx].detach().cpu())
                bucket["val_img_names"].append(sample_names[idx])
    
    def on_validation_epoch_end(self):
        top1_total = 0
        top5_total = 0
        total_sketches = 0

        for _, bucket in self.val.items():
            if not bucket["val_img_features"] or not bucket["val_sk_features"]:
                continue

            rank = torch.full((len(bucket["val_sk_names"]),), float("inf"))
            val_img_feature = torch.stack(bucket["val_img_features"])

            for idx, sketch_feature in enumerate(bucket["val_sk_features"]):
                sketch_name = bucket["val_sk_names"][idx]
                sketch_query_name = sketch_name.rsplit("/", 1)[-1]
                sketch_query_name = sketch_query_name.rsplit("\\", 1)[-1]
                sketch_query_name = sketch_query_name.rsplit(".", 1)[0]
                sketch_query_name = sketch_query_name.rsplit("-", 1)[0]

                if sketch_query_name not in bucket["val_img_names"]:
                    continue

                position_query = bucket["val_img_names"].index(sketch_query_name)

                distance = self.distance_fn(sketch_feature.unsqueeze(0), val_img_feature)
                target_distance = self.distance_fn(
                    sketch_feature.unsqueeze(0),
                    val_img_feature[position_query].unsqueeze(0),
                )

                rank[idx] = distance.le(target_distance).sum()

            top1_total += rank.le(1).sum().item()
            top5_total += rank.le(5).sum().item()
            total_sketches += rank.shape[0]

        if total_sketches == 0:
            self.val.clear()
            return

        acc1 = top1_total / total_sketches
        acc5 = top5_total / total_sketches

        self.log("acc1", acc1, on_step=False, on_epoch=True)
        self.log("acc5", acc5, on_step=False, on_epoch=True)
        self.best_metric = max(self.best_metric, acc1)

        print(f"Acc@1: {acc1:.4f}, Acc@5: {acc5:.4f}")
        self.val.clear()
