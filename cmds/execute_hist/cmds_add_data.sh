# feature extractor training
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed 10 --additional_dataset None

######################## sampling IN900 #######################

# inference on sampling dataset (no Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet-900 --inference_dataset_split ImageNet_900_train \
--inference_repeat_time 1 --inference_noAug True

# inference on sampling dataset (no Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet-100 --inference_dataset_split imageNet_100_LT_train \
--inference_repeat_time 1 --inference_noAug True

# inference on sampling dataset (w/ Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet-900 --inference_dataset_split ImageNet_900_train \
--inference_repeat_time 10

# sampling 10K at Imagenet900
bash ./cmds/shell_scrips/sampling.sh --pretrain_seed 10

seed=10
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed ${seed} \
--additional_dataset imagenet-900 \
--additional_split MAK/optimize_0.5lossR10_nearK10_coreR1.5_num10000_s${seed} \
--additional_split_name optimize_0.5lossR10_nearK10_coreR1.5_num10000_s${seed}

# summery results
python exp_analyse.py --exp_prefix "imageNet_100_LT_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch1000_batch256_s{seed}_addImagenet900optimize_0.5lossR10_nearK10_coreR1.5_num10000_s{seed}" \
--seeds 10 20 30 --dataset imagenet-100 --fewShot

# sampling 20K at Imagenet900
bash ./cmds/shell_scrips/sampling.sh --sampling_num 20000 --pretrain_seed 10

seed=10
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed ${seed} \
--additional_dataset imagenet-900 \
--additional_split MAK/optimize_0.5lossR_nearK10_coreR1.5_num20000_s${seed} \
--additional_split_name optimize_0.5lossR_nearK10_coreR1.5_num20000_s${seed}

# summery results
python exp_analyse.py --exp_prefix "imageNet_100_LT_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch1000_batch256_s{seed}_addImagenet900optimize_0.5lossR_nearK10_coreR1.5_num20000_s{seed}" \
--seeds 10 20 30 --dataset imagenet-100 --fewShot

# ablation 10K only proximity at Imagenet900
seed=10
bash ./cmds/shell_scrips/sampling.sh --pretrain_seed ${seed} --core_sample_ratio 1.0 --loss_weight 0.0
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed ${seed} \
--additional_dataset imagenet-900 \
--additional_split MAK/optimize_0.0lossR_nearK10_coreR1.0_num10000_s${seed} \
--additional_split_name optimize_0.0lossR_nearK10_coreR1.0_num10000_s${seed}

# ablation 10K loss + proximity at Imagenet900
seed=10
bash ./cmds/shell_scrips/sampling.sh --pretrain_seed ${seed} --core_sample_ratio 1.0 --loss_weight 0.5
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed ${seed} \
--additional_dataset imagenet-900 \
--additional_split MAK/optimize_0.5lossR_nearK10_coreR1.0_num10000_s${seed} \
--additional_split_name optimize_0.5lossR_nearK10_coreR1.0_num10000_s${seed}

######################## sampling imagenet_places365_mix #######################

# inference on sampling dataset (no Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet_places365_mix --inference_dataset_split imageNet_200_places_200_mix \
--inference_repeat_time 1 --inference_noAug True

# inference on training dataset (no Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet-100 --inference_dataset_split imageNet_100_LT_train \
--inference_repeat_time 1 --inference_noAug True

# inference on sampling dataset (w/ Aug)
bash ./cmds/shell_scrips/imagenet-100-inference.sh -p 5555 --workers 10 --pretrain_seed 10 \
--epochs 1000 --batch_size 256 --inference_dataset imagenet_places365_mix --inference_dataset_split imageNet_200_places_200_mix \
--inference_repeat_time 5

# sampling 10K at imagenet_places365_mix
seed=10
bash ./cmds/shell_scrips/sampling.sh --inference_dataset imagenet_places365_mix \
--inference_dataset_split imageNet_200_places_200_mix --pretrain_seed ${seed}
bash ./cmds/shell_scrips/imagenet-100-add-data.sh -g 2 -p 4866 -w 10 --seed ${seed} \
--additional_dataset imagenet_places365_mix \
--additional_split MAK/optimize_0.5lossR_nearK10_coreR1.5_num10000_s${seed} \
--additional_split_name optimize_0.5lossR_nearK10_coreR1.5_num10000_s${seed}

# summery results
python exp_analyse.py --exp_prefix "imageNet_100_LT_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch1000_batch256_s{seed}_addImagenetPlacesMixoptimize_0.5lossR_nearK10_coreR1.5_num10000_s{seed}" \
--seeds 10 20 30 --dataset imagenet-100 --fewShot
