import os
import json
from PIL import Image, ImageFile
from io import BytesIO

import requests
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words_per_cap=30, is_train=True):
        with open(ann_file, 'r') as f:
            self.ann = json.load(f)
        self.transform = transform
        self.root_path = root_path
        self.max_words_per_cap = max_words_per_cap
        self.ann_new = []

        for each in self.ann:
            image_id = each['image_id']
            filename = each['image']
            coco_url = each['coco_url']
            captions = [cap.lower() for cap in each['captions']]
            
            if is_train:
                for caption in captions:
                    self.ann_new.append({'image_id': image_id, 'filename': filename, 'coco_url': coco_url, 'caption': caption, 'captions': captions})
            else:
                self.ann_new.append({'image_id': image_id, 'filename': filename, 'coco_url': coco_url, 'caption': captions[0], 'captions': captions})
        self.ann = self.ann_new
        del self.ann_new      
        
    def __len__(self):
        return len(self.ann)

    def _download_image(self, url, save_path):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image.save(save_path)
                print(f"Image saved successfully to {save_path}")
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __getitem__(self, index):    
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image_id']

        image_path = os.path.join(self.root_path, ann['filename'])
        if not os.path.exists(image_path):
            self._download_image(ann['coco_url'], os.path.join(self.root_path, ann['filename']))
        else:
            print(f"Image already exists at {image_path}")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, caption, image_id, ann['captions']