import chainer
from chainercv.links.model.ssd import multibox_loss


class SSDClassifier(chainer.Chain):
    def __init__(self, predictor, lossfun=multibox_loss, alpha=1.0, k=3):
        super(SSDClassifier, self).__init__()
        self.alpha = alpha
        self.k = k
        self.lossfun = lossfun
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        predicted_loc, predicted_class = self.predictor(args[0])
        hat_loc, hat_class = args[1:]
        loc_loss, class_loss = self.lossfun(predicted_loc, predicted_class, hat_loc, hat_class, self.k)
        loss = loc_loss * self.alpha + class_loss
        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': class_loss},
            self)
        return loss
