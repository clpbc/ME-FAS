import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[: k].view(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def time_to_str(t, mode = 'min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError

def save_checkpoint(save_list, is_best, model, optimizer, scheduler, filename = '_checkpoint.pt'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'valid_arg': valid_args,
        'best_model_EER': best_model_HTER,
        'best_model_ACER': best_model_ACER,
        'best_model_ACC': best_model_ACC,
        'threshold': threshold
    }

    if is_best:
        torch.save(state, filename)

    def zero_param_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

