


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, flip_back_coord
from utils.transforms import transform_preds
from utils.vis import save_debug_images
from core.loss import JointsMSELoss, NMTCritierion


logger = logging.getLogger(__name__)

def trainer(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validater(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            output_x, output_y = model(input)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_coord(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_coord(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.REDUCTION_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.REDUCTION_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0