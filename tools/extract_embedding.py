#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import torch
import torchaudio
from tqdm import tqdm
import onnxruntime
import torchaudio.compliance.kaldi as kaldi


def main(args):
    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    # 配置ONNX运行时会话选项
    option = onnxruntime.SessionOptions()

    # 启用所有图优化，以提高模型推理性能
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 设置在单个操作中使用的线程数，以避免多线程操作带来的开销
    option.intra_op_num_threads = 1

    # 指定使用的执行提供者，此处为CPU执行提供者
    providers = ["CPUExecutionProvider"]

    # 创建ONNX运行时推理会话，加载指定路径的ONNX模型
    # args.onnx_path为ONNX模型文件的路径
    # sess_options传递配置选项，providers指定使用的执行提供者
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    utt2embedding, spk2embedding = {}, {}
    for utt in tqdm(utt2wav.keys()):
        # 加载音频文件及其采样率
        audio, sample_rate = torchaudio.load(utt2wav[utt])

        # 如果音频采样率不是16000Hz，则进行重采样
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

        # 计算音频的FBank特征，将音频信号进行特征提取、滤波增强高频、分帧平稳、加窗减少不连续
        feat = kaldi.fbank(audio,
                           num_mel_bins=80, #mel滤频数量。决定fbank特征的维度
                           dither=0, # 增强噪声技术， 0表示不加噪声
                           sample_frequency=16000) # 采样率

        # 对FBank特征进行均值归一化，输入正则化
        feat = feat - feat.mean(dim=0, keepdim=True)

        # 使用预训练模型计算音频的嵌入向量
        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[
            0].flatten().tolist()

        # 将嵌入向量保存到字典中，键为音频文件名
        utt2embedding[utt] = embedding

        # 获取当前音频对应的说话人
        spk = utt2spk[utt]

        # 如果说话人不在字典中，则添加该说话人
        if spk not in spk2embedding:
            spk2embedding[spk] = []

        # 将嵌入向量添加到对应说话人的列表中
        spk2embedding[spk].append(embedding)

        # 对每个说话人，计算其所有嵌入向量的平均值
        for k, v in spk2embedding.items():
            spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    torch.save(utt2embedding, '{}/utt2embedding.pt'.format(args.dir))
    torch.save(spk2embedding, '{}/spk2embedding.pt'.format(args.dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--onnx_path',
                        type=str)
    args = parser.parse_args()
    main(args)
