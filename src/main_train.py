import os
import sys
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories

def get_datasets(opts, subset_ratio=0.2):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_dataset = TrainDataset(opts)
    val_sketch = ValidDataset(opts, mode='sketch')
    val_photo = ValidDataset(opts)

    if opts.use_subset:
        train_size = int(len(train_dataset) * subset_ratio)
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = Subset(train_dataset, train_indices)
        
        val_size = int(len(val_sketch) * subset_ratio)
        val_indices = random.sample(range(len(val_sketch)), val_size)
        val_sketch = Subset(val_sketch, val_indices)
        
        val_size = int(len(val_photo) * subset_ratio)
        val_indices = random.sample(range(len(val_photo)), val_size)
        val_photo = Subset(val_photo, val_indices)


    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=val_photo, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)

    return train_loader, val_sketch_loader, val_photo_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/tuberlin", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="tuberlin", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--w_triplet", type=float, default=0.8, help="loss weight for triplet")
    parser.add_argument("--w_photo_skt", type=float, default=0.1, help="loss weight for photo-sketch contrastive (cross_loss)")
    parser.add_argument("--w_distill", type=float, default=0.1, help="loss weight for distillation")
    parser.add_argument("--w_ce", type=float, default=1.0, help="loss weight for (ce_photo + ce_sketch)")
    parser.add_argument("--w_mcc", type=float, default=0.1, help="loss weight for MCC")
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--progress', type=bool, default=False)
    parser.add_argument('--use_subset', type=bool, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio for instance-level split')
    parser.add_argument('--split_seed', type=int, default=42, help='random seed for train/val instance split')
    parser.add_argument('--fg_sbir', dest='fg_sbir', action='store_true', help='use fine-grained SBIR setup')
    parser.add_argument('--zero_shot', dest='fg_sbir', action='store_false', help='use original zero-shot class split')
    parser.set_defaults(fg_sbir=True)
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')
    
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='acc1',
        dirpath='saved_models/%s'%args.exp_name,
        filename="epoch{epoch:02d}-acc1{acc1:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        skip = [
            "model.prompt_learner_photo.token_prefix",
            "model.prompt_learner_photo.token_suffix",
            "model.prompt_learner_sketch.token_prefix",
            "model.prompt_learner_sketch.token_suffix",
        ]
        for k in skip:
            sd.pop(k, None)

        model = ZS_SBIR(args=args, classname=classnames)  # classnames = 220
        missing, unexpected = model.load_state_dict(sd, strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
