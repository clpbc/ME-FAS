# encoding: utf-8
"""
@author:  clpbc
@contact: clpzdnb@gmail.com
"""

import torch, os
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
    
def GetImgPath(rootPath, suffixs):

    imgPaths = []
    for root, dirs, files in os.walk(rootPath):
        for file in files:
            if file.endswith(tuple(suffixs)):
                imgPath = os.path.join(root, file)
                imgPaths.append(imgPath)

    return imgPaths

def ProcessOfCropSaveFace(mtcnn, imgPaths, pathOfProcessFinished):

    for imgPath in tqdm(imgPaths):
        directory, filename = os.path.split(imgPath)
        parts = directory.split(os.sep)

        img = Image.open(imgPath)
        face = mtcnn(img)

        if face is None:
            print(f'{file} is Error')

        else:
            face = face.permute(1, 2, 0).byte().numpy()
            os.makedirs(os.path.join(pathOfProcessFinished, parts[-2], parts[-1]), exist_ok = True)
            Image.fromarray(face, 'RGB').save(os.path.join(pathOfProcessFinished, parts[-2], parts[-1], filename))


if __name__ == '__main__':
    # Create face detector
    mtcnn = MTCNN(image_size = 224, device = torch.device('cuda:0'), post_process = False, margin = 30, select_largest = False)

    rootPath = r'/home/wanghaowei/cailvpan/frame/oulu'
    suffixs = ['.jpg', '.png']
    pathOfProcessFinished = r'/home/wanghaowei/cailvpan/process_frame/oulu'

    imgPaths = GetImgPath(rootPath, suffixs)
    ProcessOfCropSaveFace(mtcnn, imgPaths, pathOfProcessFinished)

