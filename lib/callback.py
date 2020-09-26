
import torch
import shutil
import os
# import ipdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val / (n + 1e-6)
        self.sum += val
        self.count += n
        self.avg = self.sum / (self.count + 1e-6)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class lr_scheduler():
    """Sets the learning rate according to the learning schedual"""

    def __init__(self, step, inter, base_lr = 0.1, factor=1, warmup=False, warmup_lr=0, warmup_step=0):

        self.step = step
        self.inter = inter
        self.base_lr = base_lr
        self.factor = factor

        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

        self.cur_epoch = 0
        self.count = 0
        self.cur_step_ind = 0
        self.lr = base_lr

    def __call__(self, optimizer, num_epoch, logger):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_epoch: int
            when the number of trained epoch equal or larger then the pre-defined number,
            change the lr value.
        """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_epoch < self.warmup_step:
            self.lr = self.warmup_lr
        elif self.step is not None:
            if self.cur_step_ind <= len(self.step)-1:
                if num_epoch >= self.step[self.cur_step_ind]:
                    self.cur_step_ind += 1
                    self.lr = self.base_lr * (self.factor ** (self.cur_step_ind))
                    # logging.info("Update[%d]: Change learning rate to %0.5e",
                    #              num_epoch, self.lr)
                    print ("Update[{:d}]: Change learning rate to {:0.5e}".format(num_epoch, self.lr))
                    logger.info("Update[{:d}]: Change learning rate to {:0.5e}".format(num_epoch, self.lr))
        else:
            tmp_lr = self.base_lr * (self.factor ** (num_epoch // self.inter))
            if tmp_lr != self.lr:
                self.lr = tmp_lr
                print ("Update[{:d}]: Change learning rate to {:0.5e}".format(num_epoch, self.lr))
                logger.info("Update[{:d}]: Change learning rate to {:0.5e}".format(num_epoch, self.lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        dir_pt = os.path.dirname(filename)
        best_pt = os.path.join(dir_pt, 'model_best.pth.tar')
        shutil.copyfile(filename, best_pt)