import os
import json
import numpy as np
import random
import time
import datetime

import torch
import ruamel.yaml as yaml
from transformers import BertTokenizer, BertModel

from dataset import create_dataset, create_loader, coco_collate_fn
from models.coco_caption_model import CocoCaptioner
from optim import create_optimizer
from scheduler import create_scheduler
import utils


def test(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, caption, image_id, captions) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        cap = model.simple_gen(image, temperature=1.0, max_length=50)
        print(cap)
        return "STOP HERE"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        del image, question_input,caption,loss 

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def evaluation(model, data_loader, tokenizer, device, config):
    pass


def cal_metric(result_file):
    pass


def save_result(result, result_dir, result_name):
    pass


def main(config):
    device = torch.device(config['device'])

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds, val_ds, test_ds = create_dataset(config)
    train_loader = create_loader(train_ds, config['batch_size_train'], 0, True, coco_collate_fn)
    val_loader = create_loader(val_ds, config['batch_size_test'], 0, False, coco_collate_fn)
    test_loader = create_loader(test_ds, config['batch_size_test'], 0, False, coco_collate_fn)

    tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_name'])
    bert_embedder = BertModel.from_pretrained(config['bert_prefix_name'])
    model = CocoCaptioner(tokenizer, bert_embedder, config)
    # optimizer = create_optimizer(config['optimizer'], model)
    # lr_scheduler, _ = create_scheduler(config['scheduler'], optimizer)
    optimizer = None
    lr_scheduler = None

    print("Start training")
    start_time = time.time()
    start_epoch = 0
    max_epoch = config['scheduler']['num_epochs']
    warmup_steps = config['scheduler']['warmup_epochs']

    for epoch in range(start_epoch, max_epoch):
        # if epoch > 0:
        #     lr_scheduler.step(epoch + warmup_steps)

        train_stats = test(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        # caption_result = evaluation(model, test_loader, tokenizer, device, config)
        # result_file = save_result(caption_result, config['result_output_dir'], 'caption_result_epoch%d' % epoch)
        # result = cal_metric(result_file)
        
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,
        #                 }
        # with open(os.path.join(config['model_output_dir'], "log.txt"), "a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

        # torch.save({
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        #     'config': config,
        #     'epoch': epoch,
        # }, os.path.join(config['model_output_dir'], 'checkpoint_%02d.pth' % epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    yaml = yaml.YAML(typ='safe')
    config = yaml.load(open('./configs/coco_config.yaml', 'r'))
    main(config)
