# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import sys, yaml, json
from datetime import datetime

sys.path.append('.')
from args import get_parser

def GetCfg():
    parser = get_parser()
    args = parser.parse_args()

    args_dict = vars(args)
    # print(json.dumps(args_dict, indent=4))

    cfgPath = args_dict['config']

    with open(cfgPath, 'r') as cfgFile:
        cfg = yaml.safe_load(cfgFile)
    
    cfg['config'] = args_dict['config']
    cfg['device'] = args_dict['device']
    cfg['ckpt'] = args_dict['ckpt']
    cfg['op_dir'] = args_dict['op_dir']

    if args_dict['source']:
        cfg['dataset']['source'] = args_dict['source']

    if args_dict['target']:
        cfg['dataset']['target'] = args_dict['target']

    if args_dict['size']:
        cfg['dataset']['size'] = args_dict['size']

    cfg['now_time'] = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg['exp_name'] = f"{cfg['model']['mode']}_{cfg['dataset']['source']}_{cfg['dataset']['target']}_{cfg['now_time']}"

    # print(json.dumps(cfg, indent=4))
    return cfg