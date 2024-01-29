import os
import random
import json

from pycocotools.coco import COCO

random.seed(42)
train_size = 0.1
train_ann_path = os.path.join(os.getcwd(), 'data', 'coco_captions', 'annotations', 'captions_train_2014.json')
test_ann_path = os.path.join(os.getcwd(), 'data', 'coco_captions', 'annotations', 'captions_test_2014.json')

ann_path = os.path.join(os.getcwd(), 'data', 'coco_captions', 'annotations', 'captions_val2014.json')
coco = COCO(ann_path)
imgIds = coco.getImgIds()
random.shuffle(imgIds)

train_imgIds = imgIds[:int(train_size*len(imgIds))]
test_imgIds = imgIds[int(train_size*len(imgIds)):]


def create_cations_ann_file(imgIds, save_ann_path=None):
    """Create a captions annotation file for a given list of coco image ids."""
    ann = []
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        coco_url = img['coco_url']
        anns = coco.loadAnns(coco.getAnnIds(imgId))
        captions = [ann['caption'] for ann in anns]
        ann.append({'image_id': imgId, 'image': filename, 'coco_url': coco_url, 'captions': captions})

    if save_ann_path is not None:
        if not os.path.exists(os.path.dirname(save_ann_path)):
            os.makedirs(os.path.dirname(save_ann_path))
        with open(save_ann_path, 'w') as f:
            json.dump(ann, f)

    return ann

train_ann = create_cations_ann_file(train_imgIds, train_ann_path)
print(f'Created train annotations file at {train_ann_path} with {len(train_ann)} images.')

test_ann = create_cations_ann_file(test_imgIds, test_ann_path)
print(f'Created test annotations file at {test_ann_path} with {len(test_ann)} images.')