import argparse
import cv2
import cvlib as cv
import pandas as pd
import os
from tqdm import tqdm

def face_detection_train():
    TRAIN_PATH = '/opt/ml/input/data/train/train_path.csv'
    WRONG_IMAGES_PATH = '/opt/ml/input/data/train/detect_test'
    RIGHT_IMAGES_PATH = '/opt/ml/input/data/train/fdimages'

    path_df = pd.read_csv(TRAIN_PATH, index_col=0)
    detect = {}
    detect['no_detect'] = 0
    detect['multi'] = 0
    detect['wrong_detect'] = 0
    detect['right_detect'] = 0
    detect['other'] = 0
    toggle = 0

    for i in tqdm(range(len(path_df))):
        if toggle == 1:
            break
        img = cv2.imread(path_df['path'][i])
        faces, confidences = cv.detect_face(img)
        img_path = path_df['path'][i].split('/')[7]
        img_name = path_df['path'][i].split('/')[8]

        if len(faces) == 0 : # no_detected
            detect['no_detect'] += 1
            img = img[50:400,20:350,:]
            save_path = os.path.join(WRONG_IMAGES_PATH, 'wrong_detect', img_path + img_name)
            cv2.imwrite(save_path,img)
            continue

        for face, conf in zip(faces, confidences) :
            left = face[0]
            top = face[1]
            right = face[2]
            bottom = face[3]
    #         try :
            if left < 384 and top < 512 and right < 384 and bottom <512 and (right-left)*(bottom-top) > 10000 and conf > 0.85:

                detect['right_detect'] += 1
                save_path = os.path.join(RIGHT_IMAGES_PATH, img_path)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                img = img[top:bottom, left:right, :]
                cv2.imwrite(os.path.join(save_path,img_name),img)
                break

def face_detection_eval():

    # image path loader
    EVAL_INFO = '/opt/ml/input/data/eval/info.csv'
    path_df = pd.read_csv(EVAL_INFO)

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
    #for idx in range(0,path_df.shape[0]):
        
        # 이미지 불러오기
        img = cv2.imread(path_df['path'][idx])
        
        # 얼굴 찾기
        faces, confidences = cv.detect_face(img)
        
        # 이미지 저장경로
        IMAGEPATH = os.path.join(FD_EVAL_IMAGES_PATH, path_df['ImageID'][idx])
        
        # face detection을 못한 경우 원본을 저장
        if len(faces) == 0:
            img = img[50:400,20:350,:]
            cv2.imwrite(IMAGEPATH, img)
            continue

        # face crop & save 
        for (x, y, x2, y2), conf in zip(faces, confidences):
            # 이미지 내에서 bbox를 그린경우만 허용, 가로길이가 너무 짧아도 pass
            try :
                if x < 384 and y < 512 and x2 < 384 and y2 <512 and (x2-x)*(y2-y) > 10000 and conf >= 0.85:
                    # conf 50ww% 이상인 bbox
                    img = img[y:y2, x:x2, :]
                    cv2.imwrite(IMAGEPATH,img)
                else :
                    img = img[50:400,20:350,:]
                    cv2.imwrite(IMAGEPATH,img)
                    
            except:
                print(IMAGEPATH, x,y,x2,y2, conf)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train, eval directories
    parser.add_argument('--data_type', type=str, default='eval', help="data type, train or eval (default='train')")

    args = parser.parse_args()

    data_dir = args.data_type

    print("Face Detection start")
    if data_dir == 'train':
        print("for train data")
        face_detection_train()
    elif data_dir == "eval":
        print("for test data")
        face_detection_eval()

    print("complete")