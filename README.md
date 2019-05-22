# SSD implementation on Chainer
## Train
```shell
!python execute.py  \
--batchsize 32 \
--num_epochs_or_iter 100 \
--initial_lr 0.001 \
--lr_decay_rate 0.1 \
--lr_decay_epoch 40 80 
```