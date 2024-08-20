# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import yaml, os, torch, random, shutil, warnings
import numpy as np

from utils import Logger
from tools import train, GetCfg 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore', category = UserWarning)


if __name__ == '__main__':
    cfg = GetCfg()  # get all config

    expSavePath = os.path.join(cfg['op_dir'], cfg['exp_name'])
    if not os.path.exists(expSavePath):
        os.makedirs(expSavePath, exist_ok = True)
    

    ### 备份当前运行code
    rootPath = r'/home/wanghaowei/cailvpan/face_anti_spoofing/CLP'
    for item in os.listdir(rootPath):
        if item in ['op_dir', 'new_template.ipynb', 'scripts']:
            continue

        srcPath = os.path.join(rootPath, item)
        desPath = os.path.join(expSavePath, 'code', item)
        
        # 复制文件或目录
        if os.path.isdir(srcPath):
            shutil.copytree(srcPath, desPath, dirs_exist_ok = True)
        else:
            shutil.copy2(srcPath, desPath)
    
    print(f"Backup of '{srcPath}' completed to '{desPath}'.")
    ### 


    ### 保存当前config
    cfgName = f"{cfg['exp_name']}.yaml"
    cfgSavePath = os.path.join(expSavePath, cfgName)

    with open(cfgSavePath, 'w', encoding = 'utf-8') as file:
        yaml.dump(cfg, file, allow_unicode = True, default_flow_style = False)
    ###


    ### 保存当前log文件与result文件
    logName = f"{cfg['exp_name']}_log.txt"
    logSavePath = os.path.join(expSavePath, logName)
    log = Logger()
    log.open(logSavePath)

    resultName = f"{cfg['exp_name']}_result.csv"
    resultSavePath = os.path.join(expSavePath, resultName)
    with open(resultSavePath, 'a') as f:
        f.write(f"{'Run': ^10}{'HTER': ^10}{'AUC': ^10}{'TPR@FPR = 1%': ^15}\n")
    ###
    
    hter_avg, auc_avg, tpr_fpr_avg = [], [], []

    for i in range(cfg['base']['repeat_num']):
        # To reproduce results 
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
        torch.cuda.manual_seed(i)

        hter, auc, tpr_fpr = train(cfg, log)

        hter_avg.append(hter)
        auc_avg.append(auc)
        tpr_fpr_avg.append(tpr_fpr)

        with open(resultSavePath, 'a') as f:
            f.write(f'{i: ^10d}{hter: ^10.4f}{auc: ^10.4f}{tpr_fpr: ^15.4f}\n')


    hter_mean = np.mean(hter_avg)
    auc_mean = np.mean(auc_avg)
    tpr_fpr_mean = np.mean(tpr_fpr_avg)

    hter_std = np.std(hter_avg)
    auc_std = np.std(auc_avg)
    tpr_fpr_std = np.std(tpr_fpr_avg)

    with open(resultSavePath, 'a') as f:
        f.write(f"\n{'Mean': ^10}{hter_mean: ^10.4f}{auc_mean: ^10.4f}{tpr_fpr_mean: ^15.4f}\n")
        f.write(f"{'Std': ^10}{hter_std: ^10.4f}{auc_std: ^10.4f}{tpr_fpr_std: ^15.4f}")