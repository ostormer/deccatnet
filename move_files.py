import os
import shutil

# expect paths as version/file type/data_split/pathology status/
#                     reference/subset/subject/recording session/file
# e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
#                     s004_2013_08_15/00000021_s004_t000.edf

def rename_files_abnormal(in_path, delete_old=False):
    """
    changes the file
    :param path: path to the tuh_eeg_abnormal_dataset
    """
    # go to v3 edf train and eval
    path = os.path.join(in_path, 'v3.0.0','edf')
    copy_path = os.path.join(in_path, 'v4.0.0','edf')
    # get name of all files in list
    data_splits = os.listdir(path)
    # do same for both eval and train
    for data_split in data_splits:
        pathology_path = os.path.join(path,data_split)
        pathologies = os.listdir(pathology_path)
        for pathology in pathologies:
            reference_path = os.path.join(pathology_path,pathology)
            references = os.listdir(reference_path)
            for reference in references:
                subset_path = os.path.join(reference_path, reference)
                files = os.listdir(subset_path)
                # first error: add all paths
                paths = ['000/','00000021/','s004_2013_08_15/']
                temp_path = subset_path
                for p in paths:
                    temp_path = os.path.join(temp_path,p)
                    try:
                        os.mkdir(temp_path)
                    except:
                        None
                        #print('path already existing')
                # move files to new path
                for file in files:
                    old_path = os.path.join(subset_path,file)
                    new_path = os.path.join(temp_path, file)
                    #print(old_path,new_path)
                    if delete_old:
                        shutil.move(old_path, new_path)
                    else:
                        shutil.copy(old_path, new_path)

#rename_files_abnormal('../datasets/TUH/tuh_eeg_abnormal/', delete_old=True)

def rename_files(path, delete_old=False, version_edf=False):
    """

    :param path: path of the tuh_eeg_corpus file. Assume that there is no version, or edf file. If
    there is a version or edf file, set version_edf to true
    :return:
    """

    if version_edf:
        temp_path = path
    else:
        paths = ['v2.0.0/','edf/']
        temp_path = path
        for p in paths:
            subset_path = os.path.join(temp_path,p)
            try:
                os.mkdir(subset_path)
                #print('create path')
            except:
                #print(subset_path + ' already existing')
                None
            temp_path = subset_path
        path = temp_path
        # now we need to get the correct 01_tcp_ar file for each file
    files = os.listdir(path)
    temp_path = '../datasets/TUH/tuh_eeg/v3.0.0/edf/'
    for file in files:
        file_path = os.path.join(path,file)
        files_2 = os.listdir(file_path)
        for file_2 in files_2:
            file_2_path = os.path.join(file_path,file_2)
            files_3 = os.listdir(file_2_path)
            for file_3 in files_3:
                file_3_path = os.path.join(file_2_path, file_3)
                files_4 = os.listdir(file_3_path)
                for file_4 in files_4:
                    file_4_path = os.path.join(file_3_path, file_4)
                    print(temp_path)
                    reference_path = os.path.join(temp_path,file_4)
                    datasplit_path = os.path.join(reference_path,file)
                    subject_path = os.path.join(datasplit_path,file_2)
                    recording_path = os.path.join(subject_path, file_3)
                    #print(reference_path, datasplit_path, subject_path, recording_path)
                    try:
                        os.mkdir(reference_path)
                    except:
                        None
                    try:
                        os.mkdir(datasplit_path)
                    except:
                        None
                    try:
                        os.mkdir(subject_path)
                    except:
                        None
                    try:
                        os.mkdir(recording_path)
                    except:
                        None
                    # move file
                    if delete_old:
                        shutil.move(file_4_path, recording_path)

                    else:
                        shutil.copytree(file_4_path,recording_path,dirs_exist_ok=True)
        if delete_old:
            shutil.rmtree(file_path)
            #print(file_path)

#rename_files('../datasets/TUH/tuh_eeg', delete_old=True)

# tuh_eeg/downloads/tuh_eeg/v2.0.0/edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar
#datasets/TUH/tuh_eeg_corpus/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf

# expect file paths as tuh_eeg/version/file_type/reference/data_split/
#                          subject/recording session/file
# e.g.                 tuh_eeg/v1.1.0/edf/01_tcp_ar/027/00002729/
#                          s001_2006_04_12/00002729_s001.edf
# for our: s: session, bokstavene: subject, t:token, dersom så lange at delt i flere filer.




# datasets/TUH/tuh_eeg_abnormal/v3.0.0/edf/train/abnormal/01_tcp_ar/aaaaaaat_s002_t001.edf
# what we need is wether it is abnormal/normal, so this can be recording session.
# data_split is assumed to be train/eval/test
#

