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
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features_norm, photo_feature = self.get_logits(photo_tensor, classnames)
        sk_logits, sk_feature_norm, sk_feature = self.get_logits(sk_tensor, classnames, type='sketch')
        _, neg_feature, _ = self.get_logits(neg_tensor, classnames)
        
        return photo_features_norm, sk_feature_norm, photo_aug_tensor, sk_aug_tensor, \
            neg_feature, label, pos_logits, sk_logits, photo_feature, sk_feature
            
    def extract_feature(self, image, classname, type='photo'):
        _, feature, _ = self.get_logits(image, classnames=classname, type=type)
        return feature
            
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
        
        self.best_metric = 0.0
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model, clip_model_distill=clip_model_distill)
    
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        
        
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
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='photo')
            self.val_step_outputs_ph.append((feat, label))
    
    def on_validation_epoch_end(self):
        if not self.val_step_outputs_sk or not self.val_step_outputs_ph:
            return

        query_feat_all = torch.cat([feat for feat, _ in self.val_step_outputs_sk], dim=0)
        gallery_feat_all = torch.cat([feat for feat, _ in self.val_step_outputs_ph], dim=0)
        query_instance_ids = torch.cat([label for _, label in self.val_step_outputs_sk], dim=0).to(query_feat_all.device)
        gallery_instance_ids = torch.cat([label for _, label in self.val_step_outputs_ph], dim=0).to(query_feat_all.device)

        similarity = query_feat_all @ gallery_feat_all.t()
        top_k = min(5, similarity.shape[1])
        topk_indices = similarity.topk(top_k, dim=1, largest=True, sorted=True).indices
        retrieved_instance_ids = gallery_instance_ids[topk_indices]
        matches = retrieved_instance_ids.eq(query_instance_ids.unsqueeze(1))

        acc1 = matches[:, :1].any(dim=1).float().mean()
        acc5 = matches[:, :top_k].any(dim=1).float().mean()

        self.log("acc1", acc1, on_step=False, on_epoch=True)
        self.log("acc5", acc5, on_step=False, on_epoch=True)
        self.best_metric = max(self.best_metric, acc1.item())

        print(f"Acc@1: {acc1.item():.4f}, Acc@5: {acc5.item():.4f}")
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
