import re
import os
import os.path as osp
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment summary parser')
    parser.add_argument('--exp_prefix', type=str, help="the exp name (template) of the pretrained model")
    parser.add_argument('--seeds', nargs='+', required=True)
    parser.add_argument('--dataset', default='imagenet100', type=str)
    parser.add_argument('--fewShot', action='store_true', help="if summery the few-shot performance")

    return parser.parse_args()


def getStatisticsFromTxt(txtName, num_class=1000):
    statistics = [0 for _ in range(num_class)]
    with open(txtName, 'r') as f:
        lines = f.readlines()
    for line in lines:
        s = re.search(r" ([0-9]+)$", line)
        if s is not None:
            statistics[int(s[1])] += 1
    return statistics


def getAccAsimclr(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^On the best_model, test tacc is ([0-9]+\.[0-9]+)$", line)
        if groups:
            bestAcc = float(groups[1])

    return bestAcc


def getClassWiseAccAsimclr(saveDir, exp, classnum=10):
    """
    :param line:
    :param save_list:
    :return:
    """
    strList = ""
    for i in range(classnum):
        strList += " ([0-9]+\.[0-9]+)"

    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return []
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    save_list = []
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^Each class acc is{}".format(strList), line)
        if groups:
            for i in range(classnum):
                save_list.append(float(groups[i+1]))

    return save_list


def getClassWiseAccImagenet(saveDir, exp, classnum=1000):
    """
    :param line:
    :param save_list:
    :return:
    """
    strList = ""
    for i in range(classnum):
        strList += " ([0-9]+\.[0-9]+)"

    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return []
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    save_list = []
    for line in lines[-5:]:
        # set_trace()
        groups = re.match("^acc per class is{}".format(strList), line)
        if groups:
            for i in range(classnum):
                save_list.append(float(groups[i+1]))

    return save_list


def getAccImagenet(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^On the best_model, test top5 tacc is ([0-9]+\.[0-9]+)", line)
        if groups:
            bestAcc = float(groups[1])

    return bestAcc


def autoSummaryExpRes(saveDir, exps, prefix, dataset='cifar10',
                      noReturnAvg=False, returnValue=False, getInfo="Asimclr", group=3, noGroup=False):
    '''
    Args:
        saveDir: str, path to save
        exps: list of tuple: (exp)
        prefix: display prefix
        dataset: which dataset
    Returns:
    '''
    accList = []
    fullVarianceList = []
    GroupVarienceList = []
    majorAccList = []
    moderateAccList = []
    minorAccList = []
    top5AccList = []
    low5AccList = []

    for exp in exps:
        if getInfo == "Asimclr":
            bestAcc = getAccAsimclr(saveDir, exp)
        elif getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            bestAcc = getAccImagenet(saveDir, exp)
        if bestAcc < 0:
            print("miss exp {}".format(exp))
            continue

        # get major moderate minor class accuracy
        if dataset == 'Imagenet':
            currentStatistics = np.array(getStatisticsFromTxt('split/ImageNet_LT/imageNet_LT_exp_train.txt'))
        elif dataset == 'Imagenet-100':
            currentStatistics = np.array(getStatisticsFromTxt('split/imagenet-100/imageNet_100_LT_train.txt', num_class=100))
        else:
            assert False

        if getInfo == "Asimclr":
            classWiseAcc = getClassWiseAccAsimclr(saveDir, exp, classnum=len(currentStatistics))
        elif getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            # set_trace()
            classWiseAcc = getClassWiseAccImagenet(saveDir, exp, classnum=len(currentStatistics))
        else:
            assert False

        # set_trace()
        if not classWiseAcc:
            print("miss classwise acc for {}".format(exp))
            assert False

        sortIdx = np.argsort(currentStatistics)
        idxsMajor = sortIdx[len(currentStatistics) // 3 * 2:]
        idxsModerate = sortIdx[len(currentStatistics) // 3 * 1: len(currentStatistics) // 3 * 2]
        idxsMinor = sortIdx[: len(currentStatistics) // 3 * 1]

        # set_trace()

        classWiseAcc = np.array(classWiseAcc)
        if getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            classWiseAcc = classWiseAcc * 100
            print("classWiseAcc is {}".format(classWiseAcc))

        bestAcc = np.mean(classWiseAcc)
        majorAcc = np.mean(classWiseAcc[idxsMajor])
        moderateAcc = np.mean(classWiseAcc[idxsModerate])
        minorAcc = np.mean(classWiseAcc[idxsMinor])

        if getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            idxsMany = np.nonzero(currentStatistics > 100)[0]
            idxsMedium = np.nonzero((100 >= currentStatistics) & (currentStatistics >= 20))[0]
            idxsFew = np.nonzero(currentStatistics < 20)[0]
            majorAcc = np.mean(classWiseAcc[idxsMany])
            moderateAcc = np.mean(classWiseAcc[idxsMedium])
            minorAcc = np.mean(classWiseAcc[idxsFew])

        accList.append(bestAcc)
        majorAccList.append(majorAcc)
        moderateAccList.append(moderateAcc)
        minorAccList.append(minorAcc)
        # balancenessList.append(imbalance_metric(classWiseAcc / 100, sigma=1))
        # print("classWiseAcc is {}".format(classWiseAcc))
        fullVarianceList.append(np.std(classWiseAcc / 100))
        GroupVarienceList.append(np.std(np.array([majorAcc, moderateAcc, minorAcc]) / 100))

        if group > 3:
            assert len(classWiseAcc) % group == 0
            group_idx_list = [sortIdx[len(currentStatistics) // group * cnt: len(currentStatistics) // group * (cnt + 1)] \
                              for cnt in range(0, group)]
            group_accs = [np.mean(classWiseAcc[group_idx_list[cnt]]) for cnt in range(0, group)]
            outputStr = "{}: group accs are".format(prefix)
            for acc in group_accs:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)

    if returnValue:
        return accList, majorAccList, moderateAccList, minorAccList
    else:
        if noReturnAvg:
            outputStr = "{}: accs are".format(prefix)
            for acc in accList:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)
            if not noGroup:
                outputStr = "{}: majorAccs are".format(prefix)
                for acc in majorAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: moderateAccs are".format(prefix)
                for acc in moderateAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: minorAccs are".format(prefix)
                for acc in minorAccList:
                    outputStr += " {:.02f}".format(acc)
            print(outputStr)
        else:
            print("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
            if not noGroup:
                print("{}: vaiance is {:.04f}+-{:.04f}".format(prefix, np.mean(fullVarianceList), np.std(fullVarianceList)))
                print("{}: GroupBalancenessList is {:.04f}+-{:.04f}".format(prefix, np.mean(GroupVarienceList), np.std(GroupVarienceList)))
                print("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
                print("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
                print("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))


def summaryIN100(exp_prefix, seeds, few_shot):
    saveDir = "checkpoints_imagenet_tune"
    exps = [exp_prefix.replace("{seed}", seed) for seed in seeds]

    if few_shot:
        exps = [exp + "__fewShot" for exp in exps]
    else:
        exps = [exp + "__lr30_wd0_epoch30_b512_d10d20" for exp in exps]

    autoSummaryExpRes(saveDir, exps, "{} ".format(exp_prefix), getInfo='Imagenet-100', dataset='Imagenet-100')


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "imagenet-100":
        summaryIN100(args.exp_prefix, args.seeds, args.fewShot)
    else:
        raise ValueError("dataset of {} is not supported, supported datasets includes [imagenet100]".format(args.dataset))
