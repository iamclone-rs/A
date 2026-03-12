import glob
import os
import random
from collections import defaultdict

import torch
from PIL import Image, ImageOps
from torchvision import transforms

DEFAULT_VAL_RATIO = 0.2
DEFAULT_SPLIT_SEED = 42

def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms


def _list_categories(root):
    sketch_root = os.path.join(root, 'sketch')
    return sorted(
        category
        for category in os.listdir(sketch_root)
        if not category.startswith('.') and os.path.isdir(os.path.join(sketch_root, category))
    )


def _photo_instance_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def _sketch_instance_id(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return stem.rsplit('-', 1)[0] if '-' in stem else stem


def _load_padded_image(filepath, max_size):
    with Image.open(filepath) as image:
        return ImageOps.pad(image.convert('RGB'), size=(max_size, max_size))


def _build_instance_records(root):
    records = []

    for category in _list_categories(root):
        photo_dir = os.path.join(root, 'photo', category)
        sketch_dir = os.path.join(root, 'sketch', category)

        photo_index = {}
        for photo_path in sorted(glob.glob(os.path.join(photo_dir, '*'))):
            if os.path.isfile(photo_path):
                photo_index[_photo_instance_id(photo_path)] = photo_path

        sketch_groups = defaultdict(list)
        for sketch_path in sorted(glob.glob(os.path.join(sketch_dir, '*'))):
            if not os.path.isfile(sketch_path):
                continue
            instance_name = _sketch_instance_id(sketch_path)
            if instance_name in photo_index:
                sketch_groups[instance_name].append(sketch_path)

        for instance_name, sketch_paths in sorted(sketch_groups.items()):
            records.append({
                'category': category,
                'instance_id': f'{category}/{instance_name}',
                'photo_path': photo_index[instance_name],
                'sketch_paths': sorted(sketch_paths),
            })

    if not records:
        raise RuntimeError(
            f'No matched sketch-photo pairs were found under {root}. '
            'Expected folders like root/photo/<category>/<image> and root/sketch/<category>/<image>-1.png.'
        )

    return records


def _split_instance_records(records, val_ratio, split_seed):
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record['category']].append(record)

    rng = random.Random(split_seed)
    train_records = []
    val_records = []

    for category in sorted(grouped_records):
        items = list(grouped_records[category])
        rng.shuffle(items)

        if len(items) == 1 or val_ratio <= 0:
            train_records.extend(items)
            continue

        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)

        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    if not val_records and len(train_records) > 1:
        rng.shuffle(train_records)
        val_records.append(train_records.pop())

    train_records.sort(key=lambda item: item['instance_id'])
    val_records.sort(key=lambda item: item['instance_id'])

    return train_records, val_records


def _get_split_records(opts):
    val_ratio = getattr(opts, 'val_ratio', DEFAULT_VAL_RATIO)
    split_seed = getattr(opts, 'split_seed', DEFAULT_SPLIT_SEED)
    records = _build_instance_records(opts.root)
    return _split_instance_records(records, val_ratio, split_seed)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opts, return_orig=False):
        self.opts = opts
        self.transform = normal_transform()
        self.return_orig = return_orig
        self.aumentation = aumented_transform()
        self.all_categories = _list_categories(self.opts.root)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.all_categories)}

        train_records, _ = _get_split_records(self.opts)

        self.samples = []
        self.photo_entries = []
        for record in train_records:
            self.photo_entries.append((record['photo_path'], record['instance_id']))
            for sketch_path in record['sketch_paths']:
                self.samples.append((
                    sketch_path,
                    record['photo_path'],
                    record['category'],
                    record['instance_id'],
                ))

        if not self.samples:
            raise RuntimeError('Training split is empty. Increase dataset size or reduce val_ratio.')

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sk_path, img_path, category, instance_id = self.samples[index]

        neg_path = img_path
        if len(self.photo_entries) > 1:
            neg_path, neg_instance_id = random.choice(self.photo_entries)
            while neg_instance_id == instance_id:
                neg_path, neg_instance_id = random.choice(self.photo_entries)

        sk_data = _load_padded_image(sk_path, self.opts.max_size)
        img_data = _load_padded_image(img_path, self.opts.max_size)
        neg_data = _load_padded_image(neg_path, self.opts.max_size)

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        sk_aug_tensor = self.aumentation(sk_data)
        img_aug_tensor = self.aumentation(img_data)
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.category_to_idx[category]


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        _, valid_records = _get_split_records(self.args)

        if not valid_records:
            raise RuntimeError('Validation split is empty. Increase dataset size or adjust val_ratio.')

        instance_to_idx = {
            record['instance_id']: idx for idx, record in enumerate(valid_records)
        }

        self.samples = []
        if self.mode == 'photo':
            for record in valid_records:
                self.samples.append((record['photo_path'], instance_to_idx[record['instance_id']]))
        else:
            for record in valid_records:
                instance_idx = instance_to_idx[record['instance_id']]
                for sketch_path in record['sketch_paths']:
                    self.samples.append((sketch_path, instance_idx))

    def __getitem__(self, index):
        filepath, instance_idx = self.samples[index]
        image = _load_padded_image(filepath, self.args.max_size)
        image_tensor = self.transform(image)
        
        return image_tensor, instance_idx
    
    def __len__(self):
        return len(self.samples)
