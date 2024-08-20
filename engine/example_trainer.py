# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch, os
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

from .example_inference import do_eval
from utils import AverageMeter, time_to_str, accuracy, save_checkpoint

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        criterion,
        device,
        log,
        epoch,
        iter_num_start
):
    best_ACC = 0.0
    best_HTER = 1.0
    best_ACER = 1.0
    best_AUC = 0.0
    best_TPR_FPR = 0.0

    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    loss_simclr = AverageMeter()
    loss_l2 = AverageMeter()
    loss_total = AverageMeter()
    classifer_top1 = AverageMeter()

    log.write(f"{'-' * 55} [START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {'-' * 55}\n\n", is_file = True)
    log.write(f"{'** start training model! **':^135}\n", is_file = True)

    message = (
        f"|{'-' * 7}|"
        f"{' VALID ':-^32}|"
        f"{' Train ':-^58}|"
        f"{' Current Best ':-^24}|"
        f"{'-' * 14}|\n"
    )
    log.write(message, is_file = True)

    message = (
        f"|{'epoch':^7}|"
        f"{'loss':^8}{'top-1':^8}{'HTER':^8}{'AUC':^8}|"
        f"{'lr':^10}{'cls-loss':^10}{'Sim-loss':^10}{'l2-loss':^9}{'total-loss':^12}{'top-1':^7}|"
        f"{'top-1':^8}{'HTER':^8}{'AUC':^8}|"
        f"{'time':^14}|\n"
    )
    log.write(message, is_file = True)

    log.write(f"|{'-' * 139}|\n", is_file = True)


    ### 训练过程
    train_iter = iter(train_loader)
    iter_per_epoch_train = len(train_iter)

    start = timer()
    for iter_num in range(iter_num_start, cfg['train']['iters'] + 1):
        if (iter_num % iter_per_epoch_train == 0):
            train_iter = iter(train_loader)

        if (iter_num != 0 and iter_num % cfg['train']['iter_per_epoch'] == 0):
            epoch += 1
        
        model.train(True)

        ###### data prepare ######
        img, aug1_img, aug2_img, label = train_iter.next()
        img, aug1_img, aug2_img, label = img.to(device), aug1_img.to(device), aug2_img.to(device), label.to(device)

        ###### forward ######
        with torch.cuda.amp.autocast(enabled = scaler is not None):
            logits, aug1_feat, aug2_feat, aug1_text_dot, aug2_text_dot = model(img, aug1_img, aug2_img, label)#, 'train')
            cls_loss = criterion['loss_1'](logits, label)
            simclr_loss = criterion['loss_2'](aug1_feat, aug2_feat)
            mse_loss = criterion['loss_3'](aug2_text_dot, aug1_text_dot)

            loss = cfg['loss_1']['weight'] * cls_loss + cfg['loss_2']['weight'] * simclr_loss + cfg['loss_3']['weight'] * mse_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        loss_classifier.update(cls_loss.item())
        loss_simclr.update(simclr_loss.item())
        loss_total.update(loss.item())

        acc = accuracy(logits, label, topk = (1, ))
        classifer_top1.update(acc[0].item())

        if (iter_num != 0 and (iter_num + 1) % cfg['train']['iter_per_epoch'] == 0):
            valid_args = do_eval(val_loader, model, device, log)

            # judge model according to HTER
            is_best = valid_args[3] <= best_HTER
            best_HTER = min(valid_args[3], best_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_HTER):
                best_ACC = valid_args[6]
                best_AUC = valid_args[4]
                best_TPR_FPR = valid_args[-1]

            save_list = [epoch, valid_args, best_HTER, best_ACC, best_ACER, threshold]
            # save_checkpoint(save_list, is_best, model, optimizer, scheduler, os.path.join(cfg['op_dir'], cfg['exp_name'], "best_checkpoint.pt"))

            print('\r', end = '', flush = True)

            message = (
                f"|{epoch:^7d}|"
                f"{valid_args[0]:^8.2f}{valid_args[6]:^8.2f}{valid_args[3] * 100:^8.2f}{valid_args[4] * 100:^8.2f}|"
                f"{lr:^10.6f}{loss_classifier.avg:^10.2f}{loss_simclr.avg:^10.2f}{loss_l2.avg:^9.2f}{loss_total.avg:^12.2f}{classifer_top1.avg:^7.2f}|"
                f"{float(best_ACC):^8.2f}{float(best_HTER * 100):^8.2f}{float(best_AUC * 100):^8.2f}|"
                f"{time_to_str(timer() - start, 'sec'):^14}|\n"
            )
            log.write(message, is_file = True)

    return best_HTER*100.0, best_AUC*100.0, best_TPR_FPR*100.0