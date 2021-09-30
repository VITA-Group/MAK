# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
# general
  -e|--epochs) pretrain_epochs="$2"; shift; shift ;;
  -s|--split) pretrain_split="$2"; shift; shift ;;
  -w|--workers) workers="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM=("$2"); shift; shift ;;
  --batch_size) batch_size=("$2"); shift; shift ;;
  --model) model=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --seed) seed=("$2"); shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
# for adding data
  --additional_dataset) additional_dataset=("$2"); shift; shift ;;
  --additional_split) additional_split=("$2"); shift; shift ;;
  --additional_split_name) additional_split_name=("$2"); shift; shift ;;
  --additional_dataset_root) additional_dataset_root=("$2"); shift; shift ;;
# strength
  --aug_strength) aug_strength=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

pretrain_epochs=${pretrain_epochs:-1000}
pretrain_split=${pretrain_split:-imageNet_100_LT_train}
batch_size=${batch_size:-256}
workers=${workers:-5}
model=${model:-res50}
dataset=${dataset:-imagenet-100}
seed=${seed:-1}
port=${port:-4833}

additional_dataset=${additional_dataset:-None}
aug_strength=${aug_strength:-"1.0"}

if [[ ${additional_dataset} == "imagenet_places365_mix" ]]
then
  additional_split=${additional_split:-imageNet_200_places_200_mix}
  additional_dataset_root=${additional_dataset_root:-"placeholder"}
  additional_split_name=${additional_split_name:-10000}
  # sampling
  sampling_dataset=${sampling_dataset:-"imagenet_places365_mix"}
  sampling_split=${sampling_split:-imageNet_200_places_200_mix}
elif [[ ${additional_dataset} == "imagenet-900" ]]
then
  additional_split=${additional_split:-imageNet_900_sub_10000}
  additional_dataset_root=${additional_dataset_root:-"placeholder"}
  additional_split_name=${additional_split_name:-10000}
  # sampling
  sampling_dataset=${sampling_dataset:-"imagenet-900"}
  sampling_split=${sampling_split:-ImageNet_900_train}
elif [[ ${additional_dataset} == "None" ]]
then
  additional_split=${additional_split:-None}
else
  echo "$dataset ${additional_dataset} is not found"; exit 125;
fi

save_dir_prefix=.
launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
save_dir=checkpoints_imagenet


if [[ ${additional_dataset} == "imagenet-900" ]]
then
  pretrain_name=${pretrain_split}_${model}_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${batch_size}_s${seed}_addImagenet900${additional_split_name}
elif [[ ${additional_dataset} == "imagenet_places365_mix" ]]
then
  pretrain_name=${pretrain_split}_${model}_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${batch_size}_s${seed}_addImagenetPlacesMix${additional_split_name}
elif [[ ${additional_dataset} == "None" ]]
then
  pretrain_name=${pretrain_split}_${model}_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${batch_size}_s${seed}
else
  echo "${additional_dataset} is not found"; exit 125;
fi

if [[ ${aug_strength} != "1.0" ]]
then
  pretrain_name="${pretrain_name}_augS${aug_strength}"
fi

cmd="${launch_cmd} train_simCLR.py \
${pretrain_name} --epochs ${pretrain_epochs} \
--batch_size ${batch_size} --output_ch 128 --lr 0.5 --temperature 0.2 --model ${model} \
--dataset ${dataset} --trainSplit ${pretrain_split} --save-dir ${save_dir} --optimizer sgd \
--num_workers ${workers} --seed ${seed} --resume"

if [[ ${additional_dataset} != "None" ]]
then
  cmd="${cmd} --additional_dataset ${additional_dataset} --additional_dataset_root ${additional_dataset_root} --additional_dataset_split ${additional_split}"
fi

if [[ ${aug_strength} != "1.0" ]]
then
  cmd="${cmd} --strength ${aug_strength}"
fi

tuneLr=30
cmd_full="${launch_cmd} \
train_imagenet.py ${pretrain_name}__lr${tuneLr}_wd0_epoch30_b512_d10d20 \
--decreasing_lr 10,20 --weight-decay 0 --epochs 30 --lr ${tuneLr} --batch-size 512 \
--model ${model} --fullset --save-dir ${save_dir_prefix}/checkpoints_imagenet_tune --dataset ${dataset} \
--checkpoint ${save_dir_prefix}/checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --num_workers ${workers} --test_freq 2"

if [[ ${dataset} == "imagenet-100" ]]
then
  train_split=imageNet_100_sub_balance_train_0.01
else
  train_split=imageNet_sub_balance_train_0.1
fi

tuneLr=30

cmd_few_shot="${launch_cmd} \
train_imagenet.py ${pretrain_name}__fewShot \
--decreasing_lr 40,60 --weight-decay 0 --epochs 100 --lr ${tuneLr} --batch-size 512 \
--model ${model} --save-dir ${save_dir_prefix}/checkpoints_imagenet_tune --dataset ${dataset} --customSplit ${train_split} \
--checkpoint ${save_dir_prefix}/checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --num_workers ${workers} --test_freq 10"

mkdir -p ${save_dir}/${pretrain_name}

echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd}
${cmd}

echo ${cmd_full} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd_full}
${cmd_full}

echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd_few_shot}
${cmd_few_shot}

