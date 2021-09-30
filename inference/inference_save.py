import time
import torch
from torch import nn
from utils import AverageMeter
import numpy as np
from os.path import join
import time
from utils import gather_tensor
from pdb import set_trace


def inference_save(inference_loader, model, log, local_rank, world_size, args, repeat_time,
                   inference_loader_txt, save_dir):

    end = time.time()

    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    separate_save = False
    if len(inference_loader) > 20000:
        log.info("separate save for large files")
        separate_save = True
        batch_cnt = 0

    with torch.no_grad():
        # separate the fc layer
        model.eval()
        fc = model.module.fc
        model.module.fc = nn.Identity()

        for cnt_run in range(repeat_time):
            features_all = []
            features_before_proj_all = []
            idxs_all = []
            labels_all = []

            for i, (inputs, labels, idxs) in enumerate(inference_loader):
                data_time = time.time() - end
                data_time_meter.update(data_time)

                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                idxs = idxs.cuda(non_blocking=True)

                features_before_proj = model(inputs)
                features = fc(features_before_proj)

                # gather idxs
                idxs = gather_tensor(idxs, local_rank, world_size)
                # gather features
                features_before_proj = gather_tensor(features_before_proj, local_rank, world_size)
                features = gather_tensor(features, local_rank, world_size)
                labels = gather_tensor(labels, local_rank, world_size)

                if local_rank == 0:
                    features_before_proj_all.append(features_before_proj.detach().cpu())
                    features_all.append(features.detach().cpu())
                    idxs_all.append(idxs.detach().cpu())
                    labels_all.append(labels.detach().cpu())

                train_time = time.time() - end
                end = time.time()
                train_time_meter.update(train_time)

                # torch.cuda.empty_cache()
                if i % args.print_freq == 0 or i == len(inference_loader) - 1:
                    log.info('Run: [{0}][{1}/{2}]\t'
                             'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                             'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                        cnt_run, i, len(inference_loader),
                        data_time=data_time_meter, train_time=train_time_meter))

                if separate_save and len(idxs_all) >= 10000:
                    features_all = torch.cat(features_all, dim=0).cpu()
                    features_before_proj_all = torch.cat(features_before_proj_all, dim=0).cpu()
                    idxs_all = torch.cat(idxs_all, dim=0).cpu().numpy()
                    labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                    torch.save(features_all, join(save_dir, "features_all_time{}_batch{}.pt".format(cnt_run, batch_cnt)))
                    if not args.inference_no_save_before_proj_feature:
                        torch.save(features_before_proj_all,
                                   join(save_dir, "features_before_proj_all_time{}_batch{}.pt".format(cnt_run, batch_cnt)))
                    np.save(join(save_dir, "idxs_all_time{}_batch{}".format(cnt_run, batch_cnt)), idxs_all)
                    np.save(join(save_dir, "labels_all_time{}_batch{}".format(cnt_run, batch_cnt)), labels_all)

                    batch_cnt += 1
                    features_all = []
                    features_before_proj_all = []
                    idxs_all = []
                    labels_all = []

            # calculate prob
            if local_rank == 0 and not separate_save:
                features_all = torch.cat(features_all, dim=0).cpu()
                features_before_proj_all = torch.cat(features_before_proj_all, dim=0).cpu()
                idxs_all = torch.cat(idxs_all, dim=0).cpu().numpy()
                labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                torch.save(features_all, join(save_dir, "features_all_time{}.pt".format(cnt_run)))
                if not args.inference_no_save_before_proj_feature:
                    torch.save(features_before_proj_all, join(save_dir, "features_before_proj_all_time{}.pt".format(cnt_run)))
                np.save(join(save_dir, "idxs_all_time{}".format(cnt_run)), idxs_all)
                np.save(join(save_dir, "labels_all_time{}".format(cnt_run)), labels_all)

            torch.distributed.barrier()

    return
