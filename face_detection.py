import argparse
import cv2
import cvlib as cv
import pandas as pd
import os
from tqdm import tqdm
import warnings



def train(data_size):
    # image path loader
    TRAIN_PATH = '/opt/ml/input/data/train/train_path.csv'
    path_df = pd.read_csv(TRAIN_PATH, index_col=0)

    # Face dection folder 
    FD_IMAGES_PATH = '/opt/ml/input/data/train/fdimages/'

    if not os.path.exists(FD_IMAGES_PATH):
        os.mkdir(FD_IMAGES_PATH)

        # 모든 이미지를 한장씩 detection
    for idx in tqdm(range(0,path_df.shape[0])):
        
        # 이미지 불러오기
        img = cv2.imread(path_df['path'][idx])
        
        # Face dection, 각 사람별 폴더 생성
        PATH = path_df['path'][idx].split('/')
        FD_IMAGEFODLER_PATH = os.path.join(FD_IMAGES_PATH, PATH[7])
        
        if not os.path.exists(FD_IMAGEFODLER_PATH):
            os.mkdir(FD_IMAGEFODLER_PATH)
        
        # 이미지 경로 지정
        IMAGEPATH = os.path.join(FD_IMAGEFODLER_PATH, PATH[8])
        
        # 얼굴 찾기
        faces, confidences = cv.detect_face(img)
        
        # face detection을 못한 경우 원본을 저장
        if len(faces) == 0:
            cv2.imwrite(IMAGEPATH, img)
            continue

        # face crop & save
        for (x, y, x2, y2), conf in zip(faces, confidences):
            # 이미지 내에서 bbox를 그린경우만 허용, 가로길이가 너무 짧아도 pass
            if x < 384 and y < 512 and x2 < 384 and y2 <512 and (x2-x)*(y2-y) > data_size:
                # conf 85% 이상인 bbox
                if conf > 0.85:
                    img = img[y:y2, x:x2, :]
                    cv2.imwrite(IMAGEPATH,img)

def eval(data_size):

    # image path loader
    EVAL_PATH = '/opt/ml/input/data/eval/info.csv'
    path_df = pd.read_csv(EVAL_PATH)

    # Face dection folder 
    EVAL_IMAGES_PATH = '/opt/ml/input/data/eval/images'

    # FD image path
    FD_EVAL_IMAGES_PATH = '/opt/ml/input/data/eval/fdimages'

    # path 열에 경로 저장
    path_df['path'] = path_df['ImageID'].apply(lambda x : os.path.join(EVAL_IMAGES_PATH, x))

    if not os.path.exists(FD_EVAL_IMAGES_PATH):
        os.mkdir(FD_EVAL_IMAGES_PATH)

        # 모든 이미지를 한장씩 detection
    for idx in tqdm(range(0,path_df.shape[0])):
        
        # 이미지 불러오기
        img = cv2.imread(path_df['path'][idx])
        
        # 얼굴 찾기
        faces, confidences = cv.detect_face(img)
        
        # 이미지 저장경로
        IMAGEPATH = os.path.join(FD_EVAL_IMAGES_PATH, path_df['ImageID'][idx])
        
        # face detection을 못한 경우 원본을 저장
        if len(faces) == 0:
            cv2.imwrite(IMAGEPATH, img)
            continue

        # face crop & save
        for (x, y, x2, y2), conf in zip(faces, confidences):
            # 이미지 내에서 bbox를 그린경우만 허용, 가로길이가 너무 짧아도 pass
            if x < 384 and y < 512 and x2 < 384 and y2 <512 and (x2-x)*(y2-y) > data_size:
                # conf 85% 이상인 bbox
                if conf > 0.85:
                    img = img[y:y2, x:x2, :]
                    cv2.imwrite(IMAGEPATH,img)



if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    # train, eval directories
    parser.add_argument('--data_type', type=str, default='train', help="data type, train or eval (default='train')")
    parser.add_argument('--size', type=int, default='10000', help="data size, width*hegiht (default='10000')")

    args = parser.parse_args()

    data_dir = args.data_type
    data_size = args.size

    print("Face Detection start")
    
    if data_dir == 'train':
        print("for train data")
        train(data_size)
    elif data_dir == "eval":
        print("for eval data")
        eval(data_size)

    print("complete")
