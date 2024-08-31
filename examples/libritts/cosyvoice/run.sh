#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts #单一语言数据集
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part} # 2014的脚本，将数据下载下来
  done
fi

#预处理数据，将数据移动到目标路径，wav、text、spk id
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir data/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx #基于onnx 使用文件夹中预训练的模型 进行 token提取
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx #基于onnx 使用预训练的模型 进行 token提取
  done
fi

# parquet格式，支持高效查询，压缩、hadoop、spark多平台支持、元数据操作索引过滤等
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# inference
 script
# 当阶段小于等于4且停止阶段大于等于4时执行推理
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # 提示用户确保tts_text中的utt存在于prompt_data中
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  # 遍历两种模式：sft和zero_shot
  for mode in sft zero_shot; do
    # 调用inference.py进行推理
    # --mode指定推理模式
    # --gpu指定使用的GPU设备
    # --config指定配置文件路径
    # --prompt_data指定prompt数据文件路径，语音数据
    # --prompt_utt2data指定utt到数据的映射文件路径，utt发音 与 data映射
    # --tts_text指定tts文本文件路径，需要合成的文本
    # --llm_model指定语言模型文件路径
    # --flow_model指定流模型文件路径
    # --hifigan_model指定HiFi-GAN模型文件路径
    # --result_dir指定结果目录路径
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test-clean/parquet/data.list \
      --prompt_utt2data data/test-clean/parquet/utt2data.list \
      --tts_text `pwd`/tts_text.json \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice/test-clean/$mode
  done
fi

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
  cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi