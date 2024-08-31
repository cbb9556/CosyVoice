# 导入必要的库
import argparse
import logging
import glob
import os
from tqdm import tqdm

# 配置日志记录器
logger = logging.getLogger()


def main():
    """
    主函数，用于处理音频文件和文本文件的映射关系，并将结果写入指定的目录。

    该函数首先查找源目录中的所有wav文件，然后为每个文件生成相应的txt文件路径。
    如果txt文件存在，则读取其内容，并根据文件名生成唯一的utterance ID和speaker ID。
    最后，将utterance与wav文件、文本内容、speaker的映射关系写入目标目录中的不同文件。
    """
    # 根据源目录路径查找所有的wav文件
    wavs = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))

    # 初始化用于存储映射关系的字典
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}

    # 遍历所有wav文件，处理并生成相应的映射关系
    for wav in tqdm(wavs):
        # 对应的txt文件路径
        txt = wav.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt):
            # 如果txt文件不存在，则记录警告并跳过当前文件
            logger.warning('{} do not exsist'.format(txt))
            continue
        # 读取txt文件内容
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())
        # 生成utterance ID
        utt = os.path.basename(wav).replace('.wav', '')
        # 生成speaker ID
        spk = utt.split('_')[0]
        # 更新映射关系
        utt2wav[utt] = wav #声波
        utt2text[utt] = content #文本内容
        utt2spk[utt] = spk #说话者ID
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    # 将映射关系写入目标目录中的文件
    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


# 配置命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()