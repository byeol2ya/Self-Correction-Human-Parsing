#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data

from matplotlib import pyplot as plt
from PIL import Image as PILImage

import networks
import utils.schp as schp
from datasets.datasets import LIPDataSet
from datasets.datasets import OURDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform, transform_parsing
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler

import wandb


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0,1,2')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()

# def visualize(edges):
#

def main():
    # args = get_arguments()
    # print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="0411-test-wild")

    start_epoch = 0
    cycle_n = 0

    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
    #     json.dump(vars(args), opt_file)

    # gpus = [int(i) for i in args.gpu.split(',')]
    # if not args.gpu == 'None':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = [256, 256]

    cudnn.enabled = True
    cudnn.benchmark = True

    print('CUDA_HOME : ', os.environ.get('CUDA_HOME'))

    # Model Initialization
    AugmentCE2P = networks.init_model('resnet101', num_classes=20) #pretrained='./pretrain_model/resnet101-imagenet.pth'
    model = DataParallelModel(AugmentCE2P).to(device) # .to(device) 원래 없었음
    # model.cuda()
    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))

    restore_from = './log/recent_addloss/checkpoint_addcycle_1.pth.tar'
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['state_dict'])
        # start_epoch = checkpoint['epoch']

    SCHP_AugmentCE2P = networks.init_model('resnet101', num_classes=20)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()

    if os.path.exists('./log/checkpoint_1_addloss.pth.tar'):
        print('Resuming schp checkpoint from {}'.format('./log/checkpoint_1_addloss.pth.tar')) # './models/exp-schp-201908261155-lip.pth'
        schp_checkpoint = torch.load('./log/checkpoint_1_addloss.pth.tar')
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = 10
        schp_model.load_state_dict(schp_model_state_dict)

    # Loss Function
    criterion = CriterionAll(lambda_1=1, lambda_2=1, lambda_3=0.1,
                             num_classes=20) # lambda_1='segmentation loss weight', lambda_2='edge loss weight', lambda_3='segmentation-edge consistency loss weight'
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # train_dataset = LIPDataSet('./datasets/LIP', 'train', crop_size=input_size, transform=transform)
    train_dataset = OURDataSet('B:/Datasets/validation', 'train', input_size=input_size, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=1,
                                   num_workers=0, shuffle=False, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=7e-3, momentum=0.9,
                          weight_decay=5e-4)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=150,
                                 eta_min=7e-3 / 100, warmup_epoch=10,
                                 start_cyclical=100, cyclical_base_lr=7e-3 / 2,
                                 cyclical_epoch=10)

    # total_iters = 20 * len(train_loader)
    start = timeit.default_timer()
    # parsing_preds = []
    # scales = np.zeros((len(train_dataset), 2), dtype=np.float32)
    # centers = np.zeros((len(train_dataset), 2), dtype=np.int32)

    ############## train starts here!
    # for epoch in range(start_epoch, 20):
    #     lr_scheduler.step(epoch=epoch)
    #     lr = lr_scheduler.get_lr()[0]
    #
    #     model.train()
    #     wandb.watch(model)
    #     for i_iter, batch in enumerate(train_loader):
    #         i_iter += len(train_loader) * epoch
    #
    #         images, labels, meta = batch
    #         images = images.permute(0, 3, 1, 2).cuda().float()
    #         # labels = labels.cuda(non_blocking=True)
    #
    #         edges = generate_edge_tensor(labels)
    #         labels = labels.type(torch.cuda.LongTensor)
    #         edges = edges.type(torch.cuda.LongTensor)
    #
    #         preds = model(images)
    #
    #         # Online Self Correction Cycle with Label Refinement
    #         if cycle_n >= 1:
    #             with torch.no_grad():
    #                 soft_preds = schp_model(images)
    #                 soft_parsing = []
    #                 soft_edge = []
    #                 for soft_pred in soft_preds:
    #                     soft_parsing.append(soft_pred[0][-1].to('cuda:0'))
    #                     soft_edge.append(soft_pred[1][-1].to('cuda:1'))
    #
    #                 soft_preds = torch.cat(soft_parsing, dim=0)
    #                 soft_edges = torch.cat(soft_edge, dim=0)
    #         else:
    #             soft_preds = None
    #             soft_edges = None
    #
    #         loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # TODO : Add 'visualize W&B with images'
    #         wandb.log({"loss": loss})
    #         if i_iter % 100 == 0:
    #             print('iter = {} of {} completed, lr = {}, loss = {}'.format(i_iter, total_iters, lr,
    #                                                                          loss.data.cpu().numpy()))
    #
    #         # wandb visualization starts
    #         # print("wandb vis start")
    #
    #         random_id = np.random.randint(0, 28)
    #
    #         # # Visualization of gt_edge in image
    #         # im = plt.imshow(edges[random_id].cpu().numpy())
    #         # wandb.log({"gt_edge":[wandb.Image(im, caption="gt_edge")]})
    #         #
    #         # # Visualization of edge prediction in image
    #         # preds_edge = ((preds[0][1])[0]).detach()
    #         # preds_edge_0 = (preds_edge[random_id, 0])
    #         # preds_edge_1 = (preds_edge[random_id, 1])
    #         # preds_ege0_image = PILImage.fromarray(preds_edge_0.cpu().numpy(), mode="L")
    #         # preds_ege1_image = PILImage.fromarray(preds_edge_1.cpu().numpy(), mode="L")
    #         # wandb.log({"preds_edge0": [wandb.Image(preds_ege0_image, caption="pred_edge0")]})
    #         # wandb.log({"preds_edge1": [wandb.Image(preds_ege1_image, caption="pred_edge0")]})
    #
    #         # Visualization of input in image
    #         input = plt.imshow(images[random_id].cpu().numpy().transpose(1,2,0))
    #         wandb.log({"input": [wandb.Image(input, caption="input")]})
    #
    #         # Visualization of gt in image
    #         gt = plt.imshow(labels[random_id].cpu().numpy())
    #         wandb.log({"gt": [wandb.Image(gt, caption="gt")]})
    #
    #         # Visualization of prediction in image
    #         pred_array = (preds[0][0][1]).detach()
    #         pred_array = (pred_array[random_id, 9])
    #         pred_array = pred_array.cpu().numpy()
    #
    #         # same type with input?
    #         wandb.log({"prediction-new": [wandb.Image(pred_array, caption="result")]})
    #
    #         # # convert grey output to rgb
    #         # image_color_map = lambda x: np.where(x >= 0,
    #         #                                      (int(255 * (x + 0.1534) / 5.4207), int(255 * (1 - x / 5.4207)), 0),
    #         #                                      (0, 0, 0))
    #         # pred_color = np.empty((64, 64, 3), dtype=np.uint8)
    #         # height, width, channels = pred_color.shape
    #         #
    #         # pred_color[:, :, 0] = pred_array
    #         # pred_color[:, :, 1] = pred_array
    #         # pred_color[:, :, 2] = pred_array
    #         #
    #         # for y in range(height):
    #         #     for x in range(width):
    #         #         for c in range(channels):
    #         #             pred_color[y, x, c] = image_color_map(pred_array[y, x])[c]
    #         #
    #         # wandb.log({"prediction": [wandb.Image(pred_color, caption="result")]})
    #
    #         # print("wandb vis finish")
    #
    #
    #     schp.save_schp_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         }, False, './log', filename='checkpoint_{}.pth.tar'.format(epoch + 1))
    #
    #     # TODO : check here!
    #     # Self Correction Cycle with Model Aggregation
    #     if (epoch + 1) >= 100 and (epoch + 1 - 100) % 10 == 0:
    #         print('Self-correction cycle number {}'.format(cycle_n))
    #         schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
    #         cycle_n += 1
    #         schp.bn_re_estimate(train_loader, schp_model)
    #         schp.save_schp_checkpoint({
    #             'state_dict': schp_model.state_dict(),
    #             'cycle_n': cycle_n,
    #         }, False, './log', filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))
    #
    #     torch.cuda.empty_cache()
    #     end = timeit.default_timer()
    #     print('epoch = {} of {} completed using {} s'.format(epoch, 20,
    #                                                          (end - start) / (epoch - start_epoch + 1)))
    ############### train finish here

    # for epoch in range(start_epoch, 20):
    with torch.no_grad():
        # lr_scheduler.step(epoch=epoch)
        # lr = lr_scheduler.get_lr()[0]

        model.eval()
        wandb.watch(model)
        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader)

            images, labels, meta = batch
            images = images.permute(0, 3, 1, 2).cuda().float()
            # labels = labels.cuda(non_blocking=True)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)

            preds = model(images)

            # # Online Self Correction Cycle with Label Refinement
            # if cycle_n >= 1:
            #     with torch.no_grad():
            #         soft_preds = schp_model(images)
            #         soft_parsing = []
            #         soft_edge = []
            #         for soft_pred in soft_preds:
            #             soft_parsing.append(soft_pred[0][-1].to('cuda:0'))
            #             soft_edge.append(soft_pred[1][-1].to('cuda:1'))
            #
            #         soft_preds = torch.cat(soft_parsing, dim=0)
            #         soft_edges = torch.cat(soft_edge, dim=0)
            # else:
            #     soft_preds = None
            #     soft_edges = None
            #
            # loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)
            # optimizer.zero_grad()
            # # loss.backward()
            # optimizer.step()
            #
            # # TODO : Add 'visualize W&B with images'
            # wandb.log({"loss": loss})
            # if i_iter % 100 == 0:
            #     print('iter = {} of {} completed, lr = {}, loss = {}'.format(i_iter, total_iters, lr,
            #                                                                  loss.data.cpu().numpy()))

            # wandb visualization starts
            # print("wandb vis start")

            random_id = np.random.randint(0, 28)

            # # Visualization of gt_edge in image
            # im = plt.imshow(edges[random_id].cpu().numpy())
            # wandb.log({"gt_edge":[wandb.Image(im, caption="gt_edge")]})
            #
            # # Visualization of edge prediction in image
            # preds_edge = ((preds[0][1])[0]).detach()
            # preds_edge_0 = (preds_edge[random_id, 0])
            # preds_edge_1 = (preds_edge[random_id, 1])
            # preds_ege0_image = PILImage.fromarray(preds_edge_0.cpu().numpy(), mode="L")
            # preds_ege1_image = PILImage.fromarray(preds_edge_1.cpu().numpy(), mode="L")
            # wandb.log({"preds_edge0": [wandb.Image(preds_ege0_image, caption="pred_edge0")]})
            # wandb.log({"preds_edge1": [wandb.Image(preds_ege1_image, caption="pred_edge0")]})

            # Visualization of input in image
            input = plt.imshow(images[0].cpu().numpy().transpose(1,2,0))
            # input = plt.imshow(images[0].cpu().numpy())
            wandb.log({"input": [wandb.Image(input, caption="input")]})

            # # Visualization of gt in image
            # gt = plt.imshow(labels[random_id].cpu().numpy())
            # wandb.log({"gt": [wandb.Image(gt, caption="gt")]})

            # Visualization of prediction in image
            pred_array = (preds[0][0][1]).detach()
            pred_array = (pred_array[0, 9])
            pred_array = pred_array.cpu().numpy()

            # same type with input?
            wandb.log({"prediction-new": [wandb.Image(pred_array, caption="result")]})
            print('how many')

            # # convert grey output to rgb
            # image_color_map = lambda x: np.where(x >= 0,
            #                                      (int(255 * (x + 0.1534) / 5.4207), int(255 * (1 - x / 5.4207)), 0),
            #                                      (0, 0, 0))
            # pred_color = np.empty((64, 64, 3), dtype=np.uint8)
            # height, width, channels = pred_color.shape
            #
            # pred_color[:, :, 0] = pred_array
            # pred_color[:, :, 1] = pred_array
            # pred_color[:, :, 2] = pred_array
            #
            # for y in range(height):
            #     for x in range(width):
            #         for c in range(channels):
            #             pred_color[y, x, c] = image_color_map(pred_array[y, x])[c]
            #
            # wandb.log({"prediction": [wandb.Image(pred_color, caption="result")]})

            # print("wandb vis finish")


        # schp.save_schp_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     }, False, './log', filename='checkpoint_{}.pth.tar'.format(epoch + 1))

        # # TODO : check here!
        # # Self Correction Cycle with Model Aggregation
        # if (epoch + 1) >= 100 and (epoch + 1 - 100) % 10 == 0:
        #     print('Self-correction cycle number {}'.format(cycle_n))
        #     schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
        #     cycle_n += 1
        #     schp.bn_re_estimate(train_loader, schp_model)
        #     schp.save_schp_checkpoint({
        #         'state_dict': schp_model.state_dict(),
        #         'cycle_n': cycle_n,
        #     }, False, './log', filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        # print('epoch = {} of {} completed using {} s'.format(epoch, 20,
        #                                                      (end - start) / (epoch - start_epoch + 1)))


    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    print("?")
    main()
    print("??")
