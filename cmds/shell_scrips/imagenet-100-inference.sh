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
  -p|--port) port=("$2"); shift; shift ;;
  --pretrain_seed) pretrain_seed=("$2"); shift; shift ;;
  --pretrain_dataset) pretrain_dataset=("$2"); shift; shift ;;
  --model) model=("$2"); shift; shift ;;
# inference settings
  --checkpoint) checkpoint=("$2"); shift; shift ;;
  --inference_repeat_time) inference_repeat_time=("$2"); shift; shift ;;
  --inference_dataset) inference_dataset=("$2"); shift; shift ;;
  --inference_dataset_split) inference_dataset_split=("$2"); shift; shift ;;
  --inference_dataset_split_name) inference_dataset_split_name=("$2"); shift; shift ;;
  --inference_noAug) inference_noAug=("$2"); shift; shift ;;
  --inference_no_save_before_proj_feature) inference_no_save_before_proj_feature=("$2"); shift; shift ;;
  --batch_size) batch_size="$2"; shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

batch_size=${batch_size:-256}
inference_noAug=${inference_noAug:-False}
inference_no_save_before_proj_feature=${inference_no_save_before_proj_feature:-False}
pretrain_seed=${pretrain_seed:-10}
model=${model:-res50}
workers=${workers:-5}
GPU_NUM=${GPU_NUM:-2}
port=${port:-4877}

# pretrain dataset
pretrain_dataset=${pretrain_dataset:-"imagenet-100"}

pretrain_epochs=${pretrain_epochs:-1000}
pretrain_split=${pretrain_split:-imageNet_100_LT_train}

# inference
inference_dataset=${inference_dataset:-"imagenet-100"}
inference_dataset_split=${inference_dataset_split:-imageNet_100_LT_train}
inference_dataset_split_name=${inference_dataset_split_name:-${inference_dataset_split}}

inference_repeat_time=${inference_repeat_time:-5}
inference_dataset_root=${inference_dataset_root:-"placeholder"}

# use ori checkpoint for reproducing
save_dir=checkpoints_imagenet

pretrain_name=${pretrain_split}_${model}_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${batch_size}_s${pretrain_seed}

if [[ ${inference_noAug} == "True" ]]
then
  pretrain_name_inference="Infer_${inference_dataset}_${inference_dataset_split_name}_noAug__${pretrain_name}"
else
  pretrain_name_inference="Infer_${inference_dataset}_${inference_dataset_split_name}__${pretrain_name}"
fi

checkpoint=${checkpoint:-${save_dir}/${pretrain_name}/model_${pretrain_epochs}.pt}

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} train_simCLR.py \
${pretrain_name_inference} --epochs ${pretrain_epochs} \
--batch_size ${batch_size} --output_ch 128 --lr 0.5 --temperature 0.2 --model ${model} \
--dataset ${pretrain_dataset} --trainSplit ${pretrain_split} --save-dir ${save_dir} --optimizer sgd \
--num_workers ${workers} --inference_dataset ${inference_dataset} --inference_dataset_root ${inference_dataset_root} \
--inference_dataset_split ${inference_dataset_split} --inference_repeat_time ${inference_repeat_time} --inference \
--checkpoint_pretrain ${checkpoint} --seed ${pretrain_seed}"

if [[ ${inference_noAug} == "True" ]]
then
  cmd="${cmd} --inference_noAug"
fi

if [[ ${inference_no_save_before_proj_feature} == "True" ]]
then
  cmd="${cmd} --inference_no_save_before_proj_feature"
fi


mkdir -p ${save_dir}/${pretrain_name}

echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd}
${cmd}
