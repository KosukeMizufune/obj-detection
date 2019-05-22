import argparse

import chainer
from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset
from chainer.datasets import TransformDataset, ConcatenatedDataset
from chainercv.links import SSD300
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.datasets import voc_bbox_label_names
from chainer import iterators

from transform import Transform
from classifier import SSDClassifier
from utils import create_trainer, trainer_extend


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epoch_or_iter', type=str, default='epoch')
    parser.add_argument('--num_epochs_or_iter', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epoch', type=float, nargs='*', default=25)
    parser.add_argument('--freeze_layer', type=str, nargs='*', default=None)
    parser.add_argument('--small_lr_layers', type=str, nargs='*', default=None)
    parser.add_argument('--small_initial_lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_trainer_interval', type=int, default=10)

    args = parser.parse_args()
    # TODO: small_lr_layers

    model = SSD300(n_fg_class=20, pretrained_model='imagenet')
    model.use_preset('evaluate')
    net = SSDClassifier(model, alpha=1, k=3)
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

    # Data Augmentation
    train = TransformDataset(
        ConcatenatedDataset(
            VOCBboxDataset(year='2007', split='trainval'),
            VOCBboxDataset(year='2012', split='trainval')
        ),
        Transform(model.coder, model.insize, model.mean)
    )
    test = VOCBboxDataset(year='2007', split='test', use_difficult=True, return_difficult=True)

    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Evaluator
    evaluator = DetectionVOCEvaluator(test_iter, net.predictor, use_07_metric=True, label_names=voc_bbox_label_names)

    trainer = create_trainer(train_iter, net, args.gpu_id, args.initial_lr,
                             args.weight_decay, args.freeze_layer, args.small_lr_layers,
                             args.small_initial_lr, args.num_epochs_or_iter,
                             args.epoch_or_iter, args.save_dir)
    if args.load_path:
        chainer.serializers.load_npz(args.load_path, trainer)

    trainer_extend(trainer, net, evaluator, args.small_lr_layers,
                   args.lr_decay_rate, args.lr_decay_epoch,
                   args.epoch_or_iter, args.save_trainer_interval)
    trainer.run()
