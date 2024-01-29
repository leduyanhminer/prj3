from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.coco_caption_dataset import coco_dataset
from dataset.randaugment import RandomAugment


def create_dataset(config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
     
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    

    train_dataset = coco_dataset(config['train_file'], train_transform, config['coco_root'], max_words_per_cap=config['max_words_per_cap'], is_train=True)
    val_dataset = coco_dataset(config['val_file'], test_transform, config['coco_root'], max_words_per_cap=config['max_words_per_cap'], is_train=False)
    test_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words_per_cap=config['max_words_per_cap'], is_train=False)
    return train_dataset, val_dataset, test_dataset  


def coco_collate_fn(batch):
    image_list, caption_list, image_id_list, captions_list = [], [], [], []
    for image, caption, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        captions_list.append(gold_caption)
    return torch.stack(image_list,dim=0), caption_list, image_id_list, captions_list


def create_loader(dataset, batch_size, num_worker, is_train, collate_fn):
    if is_train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )              
    return loader