
from Enums import EyeDiseases
import numpy as np
import pandas as pd
import os
import utilities as ut
import fileUtil


def IITD_database_checked(global_folder_path, csvpath):
    IITD_database_checked = pd.read_csv(csvpath)
    IITD_database_checked = IITD_database_checked.set_index('files')
    diseases_hand_checked, files = [], []
    for file, row in IITD_database_checked.iterrows():
        diseases = EyeDiseases.list_to_objects(row['diseases'])
        print(file)
        if row['center iris'] is np.nan or row['center pupil'] is np.nan:  # checking for pupil in iris
            continue  # if pupil in iris are not detected right we can not say that result is relevant
        diseases_hand_checked += [diseases]
        files.append(global_folder_path + file)
    return diseases_hand_checked, files


def UTIRIS_database_checked(global_folder_path, csvpath):
    UTIRIS_database_checked = pd.read_csv(csvpath)
    UTIRIS_database_checked = UTIRIS_database_checked.set_index('files')
    diseases_hand_checked, files = [], []
    for file, row in UTIRIS_database_checked.iterrows():
        diseases = EyeDiseases.list_to_objects(row['diseases'])
        print(file)
        if row['center iris'] is np.nan or row['center pupil'] is np.nan:  # checking for pupil in iris
            continue  # if pupil in iris are not detected right we can not say that result is relevant
        diseases_hand_checked += [diseases]
        files.append(global_folder_path + file)
    return diseases_hand_checked, files


def diseases_checked(path):
    d1 = f1 = d2 = f2 = d3 = f3 = d4 = f4 = []
    d1, f1 = arcus_senilis(path)
    d2, f2 = miosis(path)
    d3, f3 = mydriasis(path)
    d4, f4 = iritis_or_keratitis_pupil_shape(path)
    return d1 + d2 + d3 + d4, f1 + f2 + f3 + f4


def labeling_database(file, img_gray, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in,
                      center_pupil, radius_pupil, diseases, database):
    key = ut.draw3Circle(img_gray, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in,
                         center_pupil, radius_pupil, '/'.join(fileUtil.get_file_name(file)), label=diseases, verbose=1)
    file = file[file.index(database.value):]
    df = pd.DataFrame({'files': file, 'checked': [0], 'diseases': str(diseases), 'radius pupil': [np.nan],
                       'center pupil': [np.nan], 'radius iris': [np.nan], 'center iris': [np.nan]})
    df = df.set_index('files')
    if chr(key) == 'o':  # ok detection
        # df.loc[file, 'checked'] = '1'
        df.loc[file, "radius pupil"] = str(radius_pupil)
        df.loc[file, "center pupil"] = str(center_pupil)
        df.loc[file, "radius iris"] = str(radius_iris_out)
        df.loc[file, "center iris"] = str(center_iris_out)
    elif chr(key) == 'p':
        df.loc[file, "radius pupil"] = str(radius_pupil)
        df.loc[file, "center pupil"] = str(center_pupil)
    elif chr(key) == 'i':
        df.loc[file, "radius iris"] = str(radius_iris_out)
        df.loc[file, "center iris"] = str(center_iris_out)
    # else:  # wrong detection
    #     # df.loc[file, 'checked'] = '2'
    #     print(df.loc[file, 'diseases'])
    #     # print(df.head())
    df.to_csv(database.name + '.csv', mode='a', header=False,
              columns=['checked', 'diseases', 'radius pupil', 'center pupil', 'radius iris', 'center iris'])
    pass

def files(path, name, size):
    return [path + name + str(i + 1) + '.jpg' for i in np.arange(size)]


def arcus_senilis(path):
    name = 'eyeRing'
    if EyeDiseases.exist('ArcusSenilis'):
        diseases = [[EyeDiseases.ArcusSenilis] for d in os.listdir(path) if d.startswith(name)]
    else:
        diseases = [[] for d in os.listdir(path) if d.startswith(name)]
    diseases[5] = []
    for c in [1, 6, 7, 8, 11, 12, 13]:
        diseases[c] += [EyeDiseases.Cataract]
    for c in [0, 1, 2, 4, 7, 11, 12, 13]:
        diseases[c] += [EyeDiseases.Miosis]
    return diseases, files(path, name, len(diseases))


def miosis(path):
    name = 'miosis'
    diseases = [[EyeDiseases.Miosis] for d in os.listdir(path) if d.startswith(name)]
    for c in [3]:
        diseases[c] += [EyeDiseases.Cataract]
    return diseases, files(path, name, len(diseases))


def mydriasis(path):
    name = 'mydriasis'
    diseases = [[EyeDiseases.Mydriasis] for d in os.listdir(path) if d.startswith(name)]
    for c in [0, 1]:
        diseases[c] += [EyeDiseases.Cataract]
    return diseases, files(path, name, len(diseases))


def iritis_or_keratitis_pupil_shape(path):
    name = 'iritis'
    diseases = [[EyeDiseases.IritisOrKeratitisPupilShape] for d in os.listdir(path) if d.startswith(name)]
    diseases[1] = []
    diseases[9] = []
    diseases[2], diseases[3] = [], []
    for c in [6]:
        diseases[c] += [EyeDiseases.Mydriasis]
    for c in [3, 4, 7]:
        diseases[c] += [EyeDiseases.Miosis]
    for c in [2, 5, 7]:
        diseases[c] += [EyeDiseases.Cataract]
    return diseases, files(path, name, len(diseases))
