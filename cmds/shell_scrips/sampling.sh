# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
# general
  -e|--epochs) pretrain_epochs="$2"; shift; shift ;;
  -s|--split) pretrain_split="$2"; shift; shift ;;
  --pretrain_seed) pretrain_seed=("$2"); shift; shift ;;
  --pretrain_dataset) pretrain_dataset=("$2"); shift; shift ;;
  --model) model=("$2"); shift; shift ;;
  --batch_size) batch_size="$2"; shift; shift ;;
# inference settings
  --inference_dataset) inference_dataset=("$2"); shift; shift ;;
  --inference_dataset_split) inference_dataset_split=("$2"); shift; shift ;;
  --inference_dataset_split_name) inference_dataset_split_name=("$2"); shift; shift ;;
# sampling settings
  --K_means_num) K_means_num=("$2"); shift; shift ;;
  --loss_weight) loss_weight=("$2"); shift; shift ;;
  --core_sample_ratio) core_sample_ratio=("$2"); shift; shift ;;
  --sampling_num) sampling_num=("$2"); shift; shift ;;
  --loss_repeat_time) loss_repeat_time=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

pretrain_epochs=${pretrain_epochs:-1000}
pretrain_split=${pretrain_split:-imageNet_100_LT_train}
pretrain_seed=${pretrain_seed:-10}
pretrain_dataset=${pretrain_dataset:-"imagenet-100"}
model=${model:-res50}
batch_size=${batch_size:-256}
# inference
inference_dataset=${inference_dataset:-"imagenet-900"}
inference_dataset_split=${inference_dataset_split:-ImageNet_900_train}
inference_dataset_split_name=${inference_dataset_split_name:-${inference_dataset_split}}
# sampling
K_means_num=${K_means_num:-10}
loss_weight=${loss_weight:-0.5}
core_sample_ratio=${core_sample_ratio:-1.5}
sampling_num=${sampling_num:-10000}
loss_repeat_time=${loss_repeat_time:-10}

pretrain_name=${pretrain_split}_${model}_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${batch_size}_s${pretrain_seed}
pretrain_name_inference_no_aug="Infer_${inference_dataset}_${inference_dataset_split_name}_noAug__${pretrain_name}"
pretrain_name_inference="Infer_${inference_dataset}_${inference_dataset_split_name}__${pretrain_name}"
pretrain_name_inference_train_no_aug="Infer_${pretrain_dataset}_${pretrain_split}_noAug__${pretrain_name}"

save_dir=checkpoints_imagenet


if [[ ${inference_dataset} == "imagenet-900" ]]
then
  sampling_loader_txt="split/imagenet-900/ImageNet_900_train.txt"
  split_saving_folder="imagenet-900"
elif [[ ${inference_dataset} == "imagenet_places365_mix" ]]
then
  sampling_loader_txt="split/imagenet-places-mix/imageNet_200_places_200_mix.txt"
  split_saving_folder="imagenet-places-mix"
else
  echo "No such inference_dataset: ${inference_dataset}"; exit 125
fi

loss_txt_name="optimize_${loss_weight}lossR${loss_repeat_time}_nearK${K_means_num}_coreR${core_sample_ratio}_num${sampling_num}_s${pretrain_seed}.txt"
cmd="python sampling.py --loss_txt_name ${loss_txt_name} --loss_dist_save_dir checkpoints_sampling/${inference_dataset}_K${K_means_num}_sampling_s${pretrain_seed} \
 --split_save_dir split/${split_saving_folder}/MAK --inference_root ${save_dir} --sampling_loader_txt ${sampling_loader_txt} \
 --no_aug_sampling_dataset_inference_exp ${pretrain_name_inference_no_aug} \
 --no_aug_training_dataset_inference_exp ${pretrain_name_inference_train_no_aug} \
 --aug_sampling_dataset_inference_exp ${pretrain_name_inference} \
 --K_means_num ${K_means_num} --loss_weight ${loss_weight} --core_sample_ratio ${core_sample_ratio} \
 --sampling_num ${sampling_num} --loss_repeat_time ${loss_repeat_time}"


mkdir -p ${save_dir}/${pretrain_name}

echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd}
${cmd}
