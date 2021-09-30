import os
import sys
sys.path.append(".")
import numpy as np
from os.path import join
import torch
from utils import nt_xent_inter_batch_multiple_time, AverageMeter
from sklearn.cluster import KMeans as Kmeans_sklearn
import torch.nn.functional as F
import time
import argparse
from pdb import set_trace


def reOrderData(idxs, labels, features, features_beforeProj=None):
    # sort all losses and idxes
    labels_new = []
    features_new = []
    features_new_beforeProj = []
    idxs_new = []

    # reorder
    for cnt, (idx, label, feature) in enumerate(zip(idxs, labels, features)):
        order = np.argsort(idx)
        _, unique_idx = np.unique(idx[order], return_index=True)
        order = order[unique_idx]
        idxs_new.append(idx[order])
        labels_new.append(label[order])
        features_new.append(feature[order])
        if features_beforeProj is not None:
            features_new_beforeProj.append(features_beforeProj[cnt][order])

    # check if equal
    for cnt in range(len(idxs_new) - 1):
        if not np.array_equal(idxs_new[cnt], idxs_new[cnt+1]):
            raise ValueError("idx for {} and {} should be the same".format(cnt, cnt+1))

    if features_beforeProj is not None:
        return idxs_new, labels_new, features_new, features_new_beforeProj
    else:
        return idxs_new, labels_new, features_new


def read_data_dir(dirs, map_location="cpu", return_before_proj_features=False, limit=None):
    idxs_train, features_train, labels_train = [], [], []
    features_beforeProj = []

    cnt_file = 0

    for dir in dirs:
        print("read dir {}".format(dir))
        files = [f.replace("idxs_all_", "").replace(".npy", "") for f in os.listdir(dir) if "idxs_all_" in f]
        files = sorted(files)
        for file in files:
            cnt_file += 1
            if limit is not None and cnt_file > limit:
                break

            idxs_train.append(np.load(join(dir, "idxs_all_{}.npy".format(file))))
            features_train.append(torch.load(join(dir, "features_all_{}.pt".format(file)), map_location=map_location))
            if return_before_proj_features:
                features_beforeProj.append(torch.load(join(dir, "features_before_proj_all_{}.pt".format(file)), map_location=map_location))
            labels_train.append(np.load(join(dir, "labels_all_{}.npy".format(file))))

    if return_before_proj_features:
        idxs_new, labels_new, features_new, features_beforeProj = reOrderData(idxs_train, labels_train, features_train, features_beforeProj)
    else:
        idxs_new, labels_new, features_new = reOrderData(idxs_train, labels_train, features_train)

    if return_before_proj_features:
        return idxs_new, labels_new, features_new, features_beforeProj
    else:
        return idxs_new, labels_new, features_new


@torch.no_grad()
def cal_cloest_distance_to_feature_centroid(features_train, num_clusters, features_sample):
    features_train = torch.stack(features_train)
    features_train_avg = features_train.mean(0)
    features_avg_norm = F.normalize(features_train_avg, dim=-1).cpu().numpy()
    kmeans_ins = Kmeans_sklearn(n_clusters=num_clusters, random_state=0).fit(features_avg_norm)
    cluster_centers = torch.from_numpy(kmeans_ins.cluster_centers_)

    cluster_centers = F.normalize(cluster_centers.cuda(), dim=-1)

    # the center for each data
    features_sample = torch.stack(features_sample)
    features_sample = F.normalize(features_sample.mean(0), dim=-1)
    print("features_sample_center_list shape is {}".format(features_sample.shape))

    # calculate the distance
    closest_distance_each_sample = []
    batch_size = 2048
    steps = int(np.ceil(len(features_sample) / batch_size))
    for step in range(steps):
        features_sample_batch = features_sample[step * batch_size: (step + 1) * batch_size]
        features_sample_batch = features_sample_batch.cuda()
        # get closest distance
        closest = 1 - torch.mm(features_sample_batch, cluster_centers.t().contiguous()).max(dim=1)[0]
        closest_distance_each_sample.append(closest.detach().cpu())
    closest_distance_each_sample = torch.cat(closest_distance_each_sample).numpy()

    return closest_distance_each_sample


@torch.no_grad()
def calculateLoss_more(features_new, batch_repeat_time=10, log=None):
    train_time_meter = AverageMeter()
    # calculate loss
    losses_new = []
    pairs = [[(j, i) for i in range(j+1, len(features_new))] for j in range(0, len(features_new))]
    pairs_new = []
    for p in pairs:
        pairs_new += p

    for cnt, p in enumerate(pairs_new):
        if cnt % 10 == 0 and cnt > 1:
            msg = "run {}/{}, time avg is {:.02}s".format(cnt, len(pairs_new), train_time_meter.avg)
            if log is not None:
                log.info(msg)
            print(msg)

        end = time.time()

        i, j = p

        features, features_pair = features_new[i].cuda(), features_new[j].cuda()

        features = torch.stack([features, features_pair], dim=1).reshape(len(features) * 2, -1)
        losses = nt_xent_inter_batch_multiple_time(features, repeat_time=batch_repeat_time, t=0.2)
        losses_new.append(losses.cpu().numpy())
        train_time_meter.update(time.time() - end)

        del features
        del features_pair
        torch.cuda.empty_cache()

    # average loss
    losses_new = np.array(losses_new)
    losses_mean = losses_new.mean(0)
    return losses_mean


@torch.no_grad()
def k_core_set_greedy(inital_set, sampling_pool, sampling_num, batch_size=20000):

    train_time_meter = AverageMeter()
    with torch.no_grad():
        inital_set = F.normalize(inital_set, dim=-1)
        sampling_pool = F.normalize(sampling_pool, dim=-1)
        sampled_idxs = []
        for cnt_sampling in range(sampling_num):
            end = time.time()
            existing_samples = torch.cat([inital_set, sampling_pool[sampled_idxs]])
            excludeMask = np.ones(len(sampling_pool), dtype=np.bool)
            excludeMask[sampled_idxs] = False
            rest_samples_idx = np.nonzero(excludeMask)[0]
            cnt_batch = int(np.ceil(len(rest_samples_idx) / batch_size))
            distance_min_all = []
            for i in range(cnt_batch):
                batch_features = sampling_pool[rest_samples_idx[i * batch_size: (i + 1) * batch_size]]
                distance_min = torch.matmul(batch_features, existing_samples.t()).max(1)[0]
                distance_min_all.append(distance_min)

            distance_min_all = torch.cat(distance_min_all).detach()
            sample_idx = int(torch.argmin(distance_min_all))
            sampled_idxs.append(rest_samples_idx[sample_idx])
            train_time_meter.update(time.time() - end)
            torch.cuda.empty_cache()
            if cnt_sampling % 10 == 0:
                print("cnt_sampling is {}, time per iter is {}".format(cnt_sampling, train_time_meter.avg))
    return sampled_idxs


@torch.no_grad()
def optimize_loss_near_optimized(features_new, features_no_aug, features_train_no_aug, sampling_loader_txt,
                                 loss_weight, near_weight, save_dir, core_set_size, sampling_num=10000,
                                 write_txt_name="sampled.txt", losses=None, cloest_distance=None):

    # normalize vector
    losses = torch.FloatTensor(losses).cpu().numpy()
    losses_norm = (losses - np.mean(losses)) / np.std(losses)
    cloest_distance = torch.FloatTensor(cloest_distance).cpu().numpy()
    cloest_distance_norm = (cloest_distance - np.mean(cloest_distance)) / np.std(cloest_distance)

    # sample small in_distribution idxs
    overall_score = loss_weight * losses_norm - near_weight * cloest_distance_norm

    # sample large losses
    assert sampling_num > 0

    large_score_idx = np.argsort(overall_score)[-core_set_size:]
    print("core size is {}".format(len(large_score_idx), ))

    # performance core set choosing
    if core_set_size > sampling_num:
        sampling_idx = k_core_set_greedy(features_train_no_aug[0].cuda(), features_no_aug[0][large_score_idx].cuda(), sampling_num, batch_size=10000)
        sampling_idx = large_score_idx[sampling_idx]
    else:
        sampling_idx = large_score_idx

    # verify
    print("sampled loss with sampling idx is {}".format(np.sort(losses[sampling_idx])))
    print("sampled loss for large core set idx is {}".format(np.sort(losses[large_score_idx])))

    # sample and write
    # write txt for sampling
    with open(sampling_loader_txt, 'r') as f:
        allLines = f.readlines()

    sampledLines = np.array(allLines)[sampling_idx].tolist()
    os.system("mkdir -p {}".format(save_dir))
    with open(join(save_dir, write_txt_name), 'w') as f:
        f.writelines(sampledLines)

    return


@torch.no_grad()
def optimize_loss_near_normalized_togather_core(inference_root, split_save_dir, sampling_loader_txt,
                                                loss_txt_name, loss_dist_save_dir,
                                                no_aug_sampling_dataset_inference,
                                                no_aug_training_dataset_inference,
                                                aug_sampling_dataset_inference,
                                                K_means_num=10, loss_weight=0.5,
                                                core_sample_ratio=1.5, sampling_num=10000,
                                                loss_repeat_time=10):
    idxs_no_aug, labels_no_aug, _, features_no_aug_beforeProj = read_data_dir([join(inference_root, no_aug_sampling_dataset_inference), ],
                                                                              map_location="cpu", return_before_proj_features=True)
    idxs_train_no_aug, labels_train_no_aug, _, features_train_no_aug_beforeProj = read_data_dir([join(inference_root, no_aug_training_dataset_inference), ],
                                                                                                map_location="cpu", return_before_proj_features=True)
    idxs, labels, features = read_data_dir([join(inference_root, aug_sampling_dataset_inference),], map_location="cpu",
                                           limit=loss_repeat_time)

    os.system("mkdir -p {}".format(loss_dist_save_dir))

    if loss_repeat_time > len(features):
        raise ValueError("the inference time of {} is not enough for supporting loss_repeat_time of {}".format(len(features), loss_repeat_time))
    print("The len for features under training is {}".format(loss_repeat_time))
    if os.path.isfile("{}/{}".format(loss_dist_save_dir, "loss_{}.npy".format(loss_repeat_time))):
        losses = np.load("{}/{}".format(loss_dist_save_dir, "loss_{}.npy".format(loss_repeat_time)))
    else:
        # calculating loss
        torch.manual_seed(0)
        losses = calculateLoss_more(features)
        np.save("{}/{}".format(loss_dist_save_dir, "loss.npy"), losses)

    if os.path.isfile("{}/{}".format(loss_dist_save_dir, "distance_{}.npy".format(K_means_num))):
        cloest_distance = np.load("{}/{}".format(loss_dist_save_dir, "distance_{}.npy".format(K_means_num)))
    else:
        # calculating cloest distance
        cloest_distance = cal_cloest_distance_to_feature_centroid(features_train_no_aug_beforeProj, num_clusters=K_means_num, features_sample=features_no_aug_beforeProj)
        np.save("{}/{}".format(loss_dist_save_dir, "distance_{}.npy".format(K_means_num)), cloest_distance)

    core_sample_size = int(core_sample_ratio * sampling_num)
    optimize_loss_near_optimized(features, features_no_aug_beforeProj, features_train_no_aug_beforeProj,
                                 sampling_loader_txt, loss_weight, 1-loss_weight, split_save_dir, core_sample_size,
                                 sampling_num=sampling_num, write_txt_name=loss_txt_name, losses=losses,
                                 cloest_distance=cloest_distance)


def main():
    parser = argparse.ArgumentParser(description='PyTorch sampling')
    # saving dir
    parser.add_argument('--loss_txt_name', type=str, help='name of saved checkpoint')
    parser.add_argument('--loss_dist_save_dir', default='checkpoints_sampling/imagenet_900_sampling_s10',
                        type=str, help='path to save loss and dist for re-using')
    parser.add_argument('--split_save_dir', default='split/imagenet-900/MAK', type=str, help='path to save generated split')
    parser.add_argument('--inference_root', default='checkpoints_imagenet', type=str, help='root of inference result')
    parser.add_argument('--sampling_loader_txt', default='split/imagenet-900/ImageNet_900_train.txt', type=str, help='the inference dataset split')
    parser.add_argument('--no_aug_sampling_dataset_inference_exp', default='path to inference results on inference dataset w/o augmentation', type=str)
    parser.add_argument('--no_aug_training_dataset_inference_exp', default='path to inference results on training dataset w/o augmentation', type=str)
    parser.add_argument('--aug_sampling_dataset_inference_exp', default='path to inference results on inference dataset w/ augmentation', type=str)

    # sampling hyper-parameters
    parser.add_argument('--K_means_num', default=10, type=int)
    parser.add_argument('--loss_weight', default=0.5, type=float)
    parser.add_argument('--core_sample_ratio', default=1.5, type=float)
    parser.add_argument('--sampling_num', default=10000, type=int)
    parser.add_argument('--loss_repeat_time', default=10, type=int)

    args = parser.parse_args()
    optimize_loss_near_normalized_togather_core(args.inference_root, args.split_save_dir, args.sampling_loader_txt,
                                                args.loss_txt_name, args.loss_dist_save_dir,
                                                args.no_aug_sampling_dataset_inference_exp,
                                                args.no_aug_training_dataset_inference_exp,
                                                args.aug_sampling_dataset_inference_exp,
                                                K_means_num=args.K_means_num, loss_weight=args.loss_weight,
                                                core_sample_ratio=args.core_sample_ratio, sampling_num=args.sampling_num,
                                                loss_repeat_time=args.loss_repeat_time)


if __name__ == "__main__":
    main()
