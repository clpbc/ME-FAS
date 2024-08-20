import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='clp pytorch training and testing ')

    # base args
    parser.add_argument('--config', required = True, help = 'yaml配置文件')

    parser.add_argument('--device', default = 'cuda:0', help = '使用的设备(cpu/cuda:0/cuda:1)')
    parser.add_argument('--ckpt', type = str, default = None)
    parser.add_argument('--op_dir', type = str, default = './op_dir', help = 'checkpoint保存路径')

    parser.add_argument('--source', type = str, help = 'source dataset type')
    parser.add_argument('--target', type = str, help = 'target dataset type')
    parser.add_argument('--size', type = int, help = 'sample size of each datasetType and labelType in each batch')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
