import glob
import os
import random
from collections import defaultdict

import torch
from PIL import Image, ImageOps
from torch.utils.data import Sampler, Subset
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
        self.records = train_records
        self.instance_to_idx = {
            record['instance_id']: idx for idx, record in enumerate(self.records)
        }
        self.sample_category_indices = [
            self.category_to_idx[record['category']] for record in self.records
        ]

        if not self.records:
            raise RuntimeError('Training split is empty. Increase dataset size or reduce val_ratio.')

    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, index):
        record = self.records[index]
        sk_path = random.choice(record['sketch_paths'])
        img_path = record['photo_path']
        category_idx = self.category_to_idx[record['category']]
        instance_idx = self.instance_to_idx[record['instance_id']]

        sk_data = _load_padded_image(sk_path, self.opts.max_size)
        img_data = _load_padded_image(img_path, self.opts.max_size)

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        
        sk_aug_tensor = self.aumentation(sk_data)
        img_aug_tensor = self.aumentation(img_data)
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, category_idx, instance_idx


def _category_indices_for_sampler(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if not hasattr(base_dataset, 'sample_category_indices'):
            raise ValueError('Subset base dataset must expose sample_category_indices for PK sampling.')
        return [base_dataset.sample_category_indices[idx] for idx in dataset.indices]

    if not hasattr(dataset, 'sample_category_indices'):
        raise ValueError('Dataset must expose sample_category_indices for PK sampling.')

    return list(dataset.sample_category_indices)


class PKBatchSampler(Sampler):
    def __init__(self, dataset, classes_per_batch, instances_per_class, batches_per_epoch=None):
        self.dataset = dataset
        self.classes_per_batch = classes_per_batch
        self.instances_per_class = instances_per_class
        self.batch_size = classes_per_batch * instances_per_class

        category_indices = _category_indices_for_sampler(dataset)
        self.category_to_sample_indices = defaultdict(list)
        for sample_idx, category_idx in enumerate(category_indices):
            self.category_to_sample_indices[category_idx].append(sample_idx)

        self.available_categories = [
            category for category, indices in self.category_to_sample_indices.items() if len(indices) > 0
        ]
        self.hard_negative_categories = [
            category for category, indices in self.category_to_sample_indices.items() if len(indices) >= 2
        ] or self.available_categories

        if not self.available_categories:
            raise RuntimeError('PKBatchSampler could not find any training samples.')

        if batches_per_epoch is None:
            total_samples = len(category_indices)
            self.batches_per_epoch = max(1, total_samples // self.batch_size)
            if total_samples % self.batch_size:
                self.batches_per_epoch += 1
        else:
            self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            if len(self.hard_negative_categories) >= self.classes_per_batch:
                batch_categories = random.sample(self.hard_negative_categories, self.classes_per_batch)
            else:
                batch_categories = random.choices(self.hard_negative_categories, k=self.classes_per_batch)

            batch_indices = []
            for category in batch_categories:
                sample_indices = self.category_to_sample_indices[category]
                if len(sample_indices) >= self.instances_per_class:
                    batch_indices.extend(random.sample(sample_indices, self.instances_per_class))
                else:
                    batch_indices.extend(random.choices(sample_indices, k=self.instances_per_class))

            random.shuffle(batch_indices)
            yield batch_indices


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        _, valid_records = _get_split_records(self.args)
        self.all_categories = _list_categories(self.args.root)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.all_categories)}

        if not valid_records:
            raise RuntimeError('Validation split is empty. Increase dataset size or adjust val_ratio.')

        self.samples = []
        if self.mode == 'photo':
            for record in valid_records:
                photo_name = _photo_instance_id(record['photo_path'])
                self.samples.append((
                    record['photo_path'],
                    self.category_to_idx[record['category']],
                    photo_name,
                ))
        else:
            for record in valid_records:
                for sketch_path in record['sketch_paths']:
                    sketch_name = os.path.basename(sketch_path)
                    self.samples.append((
                        sketch_path,
                        self.category_to_idx[record['category']],
                        sketch_name,
                    ))

    def __getitem__(self, index):
        filepath, category_idx, instance_name = self.samples[index]
        image = _load_padded_image(filepath, self.args.max_size)
        image_tensor = self.transform(image)
        
        return image_tensor, category_idx, instance_name
    
    def __len__(self):
        return len(self.samples)
