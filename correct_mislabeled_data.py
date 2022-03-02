import os
from glob import glob
import pandas as pd
import shutil

def isFileChanged(file_path): # check if file changed based on last modified time 
    if os.path.getmtime(file_path) > 1614865992.0: 
        return True 

    return False

def correct_mask_status(data_dir, invalid_list, correct_list):
    print("="*50)
    print("Change Mask Status")

    for folder in invalid_list: 
        image_dir = os.path.join(data_dir, folder)

        incorrect_file = os.path.join(image_dir, correct_list[0])
        normal_file = os.path.join(image_dir, correct_list[1])

        if not isFileChanged(incorrect_file): #last modified time
            temp = os.path.join(image_dir, correct_list[2])

            os.rename(incorrect_file, temp)
            os.rename(normal_file, incorrect_file)
            os.rename(temp, normal_file)  # temp.jpg is not created in this folder

            print("Changed File Names")

        else :
            print("Already Changed")
    
    print("Process Done")



def change_incorrect_gender(incorrect, src, target): 
    changed_path = incorrect.replace(src, target)
    print(f"{incorrect.split('images/')[1]} is changing into {target}")
    shutil.move(incorrect, changed_path)


def correct_gender_status(data_dir, invalid_id_list): 
    df = pd.read_csv('/opt/ml/input/data/train/train.csv')

    path_list = [] # list that contains incorrect file path
    correct_gender_list = [] # list that contains correct gender of incorrect file path

    print("="*50)
    print("Change Gender Status")

    for path in df['path']:
        for pid in invalid_id_list:
            if pid in path:
                path_list.append(path)
                correct_gender_list.append("male" if "female" in path else "female")

    for idx, foldername in enumerate(path_list):
        folder_dir = os.path.join(data_dir, foldername)
        gender = correct_gender_list[idx]

        if gender == "male" and os.path.exists(folder_dir):
            change_incorrect_gender(folder_dir, "female", gender)

        elif gender=="female" and os.path.exists(folder_dir):
            change_incorrect_gender(folder_dir, "male", gender)

    print("Process Done")




def readCurrentFolders(data_dir): # change age task should read current folders
    return sorted(list(filter(lambda p:not p.startswith("."), os.listdir(data_dir))))    


def correct_age_status(data_dir, invalid_age_id):
    current_folders = readCurrentFolders(data_dir)
    idx = 0 # for age dict

    print("="*50)
    print("Change Age Status")

    for folder in current_folders: #sorted folders list
        invalid_path = os.path.join(data_dir, folder)
        invalid_age = folder.split("Asian_")[1]

        if folder.split("_")[0] in invalid_age_id.keys():
            correct_age = str(list(invalid_age_id.values())[idx])
            correct_path = invalid_path.replace(invalid_age, correct_age)

            print(f"{invalid_path.split('images/')[1]} is changing into {correct_path.split('images/')[1]}")
            shutil.move(invalid_path, correct_path)

            idx += 1


    print("Process Done")


if __name__ == "__main__":
    data_dir= "/opt/ml/input/data/train/images"

    mask_status_invalid = ["000020_female_Asian_50", "004418_male_Asian_20", "005227_male_Asian_22"]
    mask_status_name = ["incorrect_mask.jpg", "normal.jpg", "temp.jpg"]

    gender_status_invalid = ["000225", "000664", "000767", "001498-1", "001509", "003113", "003223", "004281", 
    "004432", "005223", "006359", "006360","006361", "006362", "006363", "006364", "006424"]

    age_status_invalid = {"001009" : 20, "001064": 20, "001637":20, "001666":20, "001852":20, "004348": 60}

    correct_mask_status(data_dir, mask_status_invalid, mask_status_name)
    correct_gender_status(data_dir, gender_status_invalid) 
    correct_age_status(data_dir, age_status_invalid)