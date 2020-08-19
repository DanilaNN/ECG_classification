import json
import os
import pathlib
import matplotlib.pyplot as plt
import pylab
from   shutil import copyfile
from   pprint import pprint
import biosppy
from biosppy.signals import ecg

dump_big_json_file                          = 0
delete_or_copy_ECG_with_needed_length       = 1
create_ECG_with_nx5000_dimension            = 0
find_and_save_ECG_NormoSystole_and_others   = 0
find_necessary_ECG_and_save_to_folders      = 0
create_test_train_folders_for_ekg           = 0
augmentation_in_folder_ReadyData_2          = 0
create_folder_with_1_beat_ekg               = 0
biospy_augmentation                         = 0
plot_ECG                                    = 0
plot_ECG_in_directory                       = 0

###############################################################################
##              Dump big json file
###############################################################################
if  dump_big_json_file:
    with open('../ecg_1078/data_1078.json') as f:
        data = json.load(f)

    #for i in data.keys():
    pprint(len(data))

    for i, val in data.items():
        with open('../Data/'+str(i) + '.json', 'w') as outfile:
            json.dump(val, outfile)


###############################################################################
##              Delete or copy ECG with needed length
###############################################################################
if delete_or_copy_ECG_with_needed_length:
    #dataPath = '../ReadyData_2class_1D/Train/regular_normosystole/'
    dataPath = '../1beat/1beat/Test/regular_normosystole/'
    copyDataPath = '../Data_2500/'

    tree = os.walk(dataPath)
    fullRoots = []
    AllFiles = []
    for root, dirs, files in tree:
        fullRoots.append(root)
        AllFiles.append(files)
    AllFiles = AllFiles[0]

    print(len(AllFiles))

    count = 0
    for file in AllFiles:
        with open(dataPath+file) as f:
            data = json.load(f)

        if(len(data['Signal']) != 300):
            os.remove(dataPath + file)
            count += 1

        #if len(data['Leads']['avf']['Signal']) != 5000:
         #   count += 1
          #  copyfile(dataPath + file, copyDataPath + file)
           # os.remove(dataPath + file)

        #if len(data['Leads']) != 12:
        #    count += 1
        #    os.remove(dataPath + file)

    print(count)


###############################################################################
##              Find necessary EKGS and save to folders
###############################################################################
if find_necessary_ECG_and_save_to_folders:
    targetKeys = ['regular_normosystole', 'sinus_tachycardia', 'sinus_bradycardia',
                'sinus_arrhythmia', 'irregular_sinus_rhythm', 'atrial_fibrillation']


    for targetKey in targetKeys:

        pathlib.Path('../EKG/'+targetKey).mkdir(parents=True, exist_ok=True)

        EKG_arr = []

        tree = os.walk('../Data/')

        files = []
        for i in tree:
            files.append(i[2])
        files = files[0]

        for file in files:
            with open('../Data/' + file) as f:
                data = json.load(f)

            try:
                if data['StructuredDiagnosisDoc'][targetKey]:
                    EKG_arr.append(file[2][0])
                    copyfile('../Data/'+file, '../EKG/'+targetKey+'/'+file)

            except KeyError:
                pass

            f.close()

        print(len(EKG_arr))


###############################################################################
##     Find ECG NormoSystole and others and save them to appropriate folders
###############################################################################
if find_and_save_ECG_NormoSystole_and_others:
    targetKey = 'regular_normosystole'
    PathSaveTo = '../EKG_2_class_1D/'
    PathSaveToOthers = PathSaveTo + 'others/'
    PathFrom = '../Data_1D/'

    pathlib.Path(PathSaveTo + targetKey).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PathSaveToOthers).mkdir(parents=True, exist_ok=True)

    EKG_arr = []

    tree = os.walk(PathFrom)

    files = []
    for i in tree:
        files.append(i[2])
    files = files[0]


    for file in files:
        with open(PathFrom + file) as f:
            data = json.load(f)

        #try:
            #if data['StructuredDiagnosisDoc'][targetKey]:
        if data['diagVec'][targetKey]:
            EKG_arr.append(file[2][0])
            copyfile(PathFrom+file, PathSaveTo+targetKey+'/'+file)
        else:
            copyfile(PathFrom + file, PathSaveToOthers + file)

        #except KeyError:
         #   pass

        f.close()

    print(len(EKG_arr))

###############################################################################
##      Create test train folders for ekg
###############################################################################
if create_test_train_folders_for_ekg:
    #pathlib.Path('../EKG/Test').mkdir(parents=True, exist_ok=True)

    EKGPath = '../EKG_2_class_1D/'
    trainDataPath = '../ReadyData_2class_1D/Train/'
    testDataPath = '../ReadyData_2class_1D/Test/'

    tree = os.walk(EKGPath)
    fullRoots = []
    AllFiles = []
    for root, dirs, files in tree:
        fullRoots.append(root)
        AllFiles.append(files)
    fullRoots = fullRoots[1:]
    AllFiles = AllFiles[1:]

    dirs = []
    for roots in fullRoots:
        dirs.append(roots.split('/')[2])

    ## create dict: {diag: EKGs arr}
    ekgDict = {}
    count = 0
    for dir in dirs:
        ekgDict.update({dir: AllFiles[count]})
        count = count + 1


    count = 0
    for key in ekgDict.keys():
        for file in ekgDict[key]:
            if count % 10 == 0:
                pathlib.Path(testDataPath + key+'/' ).mkdir(parents=True, exist_ok=True)
                copyfile(EKGPath + key+'/' + file, testDataPath + key+'/' + file)
            else:
                pathlib.Path(trainDataPath + key + '/').mkdir(parents=True, exist_ok=True)
                copyfile(EKGPath + key + '/' + file, trainDataPath + key + '/' + file)

            count = count + 1
        count = 0


###############################################################################
##      plot ekg
###############################################################################
if plot_ECG:
    EKGPath_1 = '../1beat/1beat/Train/regular_normosystole/50778881_3.json'
    EKGPath_2 = '../1beat/1beat/Train/regular_normosystole/50778881_4.json'
    EKGPath_3 = '../1beat/1beat/Train/regular_normosystole/50778881_5.json'

    #EKGPath_1 = '../ReadyData_2class_1D/Train/regular_normosystole/50778881.json'
    #EKGPath_2 = '../ReadyData_2class_1D/Train/regular_normosystole/50778925.json'
    #EKGPath_3 = '../ReadyData_2class_1D/Train/regular_normosystole/50808820.json'

    with open(EKGPath_1) as f:
        data_1 = json.load(f)

    with open(EKGPath_2) as f:
        data_2 = json.load(f)

    with open(EKGPath_3) as f:
        data_3 = json.load(f)

    #print(data['Leads']['avf']['Signal'])

    #print(len(data['Leads']['avf']['Signal']))
    #print(len(range(data['Leads']['avf']['Signal'])))

    x = range(len(data_1['Signal']))[0:5000]
    y1 = data_1['Signal']
    y2 = data_2['Signal']
    y3 = data_3['Signal']

    pylab.subplot(3, 1, 1)
    pylab.plot(x, y1)
    pylab.title("atrial_fibrillation")

    pylab.subplot(3, 1, 2)
    pylab.plot(x, y2)
    pylab.title("sinus_arrhythmia")

    pylab.subplot(3, 1, 3)
    pylab.plot(x, y3)
    pylab.title("regular_normosystole")

    #plt.plot(range(len(data_1['Signal'][0])), data_1['Signal'][0])
    plt.show()

###############################################################################
##      augmentation in folder ReadyData_2
###############################################################################
if augmentation_in_folder_ReadyData_2:
    FULL_EKG_LEN = 5000
    TAR_EKG_LEN = 2500
    EKG_AUG_STEP = 100
    mainFolderTo = '../ReadyDataAug/'
    trainFolderTo = mainFolderTo + 'Train/'
    testFolderTo = mainFolderTo + 'Test/'

    mainFolderFrom = '../ReadyData/'
    trainFolderFrom = mainFolderFrom + 'Train/'
    testFolderFrom = mainFolderFrom + 'Test/'

    pathlib.Path(mainFolderTo).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trainFolderTo).mkdir(parents=True, exist_ok=True)
    pathlib.Path(testFolderTo).mkdir(parents=True, exist_ok=True)

    def func(trainFlag):
        mainPathFrom = trainFolderFrom if trainFlag else testFolderFrom
        mainPathTo = trainFolderTo if trainFlag else testFolderTo
        tree = os.walk(mainPathFrom)
        fullRoots = []
        AllFiles = []
        for root, dirs, files in tree:
            fullRoots.append(root)
            AllFiles.append(files)
        fullRoots = fullRoots[1:]
        AllFiles = AllFiles[1:]

        i = 0
        for path in fullRoots:
            for file in AllFiles[i]:
                with open(path + '/' + file) as f:
                    data = json.load(f)

                diagnFolder = path.split('/')[3]
                pathlib.Path(mainPathTo + diagnFolder).mkdir(parents=True, exist_ok=True)

                idxStart = 0
                idxStop = TAR_EKG_LEN
                while idxStop != FULL_EKG_LEN + EKG_AUG_STEP:

                    signArr = []
                    for key in data['Leads'].keys():
                        signArr.append(data['Leads'][key]['Signal'][idxStart:idxStop])

                    dictToWrite = {'diagVec': data['StructuredDiagnosisDoc'], 'Signal': signArr}
                    newFileName = file.split('.')[0] + '_' + str(idxStart) + '.' + file.split('.')[1]
                    with open(mainPathTo + diagnFolder + '/' + newFileName, 'w') as outfile:
                        json.dump(dictToWrite, outfile)

                    idxStart += EKG_AUG_STEP
                    idxStop += EKG_AUG_STEP

            i += 1
    func(0)

###############################################################################
##              Create EKG-s with nx5000 dimension
###############################################################################
if create_ECG_with_nx5000_dimension:
    PATH_FROM = '../Data/'
    PATH_TO = '../Data_1D/'
    TARGET_SIGNAL_KEY = 'iii'
    numDim = 1

    pathlib.Path(PATH_TO).mkdir(parents=True, exist_ok=True)

    fullRoots = []
    AllFiles = []
    tree = os.walk(PATH_FROM)
    for root, dirs, files in tree:
        fullRoots.append(root)
        AllFiles.append(files)
    print(fullRoots)
    AllFiles = AllFiles[0]

    count = 0
    for file in AllFiles:
        with open(PATH_FROM + file) as f:
            data = json.load(f)

        signArr = []
        for key in data['Leads'].keys():

            if key == TARGET_SIGNAL_KEY:
                signArr.append(data['Leads'][key]['Signal'])
                count += 1
                break

        dictToWrite = {'diagVec': data['StructuredDiagnosisDoc'], 'Signal': signArr}

        with open(PATH_TO + file, 'w') as outfile:
            json.dump(dictToWrite, outfile)

    print(count)

###############################################################################
##              work with biospy
###############################################################################

if biospy_augmentation:
    import biosppy
    from biosppy.signals import ecg

    EKGPath = '../ReadyDataAug (copy)/Train/regular_normosystole/1102593484_1800.json'

    with open(EKGPath) as f:
        data_1 = json.load(f)

    data = data_1['Signal'][0]

    #out = ecg.ecg(signal=data, sampling_rate=1000., show=True)

    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
    print(peaks)

    x = range(len(data_1['Signal'][0]))[0:300]
    y1 = data_1['Signal'][0][peaks[0] - 100:peaks[0] + 200]

    pylab.subplot(3, 1, 1)
    pylab.plot(x, y1)
    pylab.title("atrial_fibrillation")

    #plt.plot(range(len(data_1['Signal'][0])), data_1['Signal'][0])
    plt.show()


###############################################################################
##              Create Folder with 1-beat ekg
###############################################################################
if create_folder_with_1_beat_ekg:
    FULL_EKG_LEN = 5000
    TAR_EKG_LEN = 300
    EKG_AUG_STEP = 100

    PEAK_MINUS = 100
    PEAK_PLUS = 200

    mainFolderTo = '../1beat/1beat/'
    trainFolderTo = mainFolderTo + 'Train/'
    testFolderTo = mainFolderTo + 'Test/'

    mainFolderFrom = '../ReadyData_2class_1D/'
    trainFolderFrom = mainFolderFrom + 'Train/'
    testFolderFrom = mainFolderFrom + 'Test/'

    pathlib.Path(mainFolderTo).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trainFolderTo).mkdir(parents=True, exist_ok=True)
    pathlib.Path(testFolderTo).mkdir(parents=True, exist_ok=True)

    def func(trainFlag):
        mainPathFrom = trainFolderFrom if trainFlag else testFolderFrom
        mainPathTo = trainFolderTo if trainFlag else testFolderTo
        tree = os.walk(mainPathFrom)
        fullRoots = []
        AllFiles = []
        for root, dirs, files in tree:
            fullRoots.append(root)
            AllFiles.append(files)
        fullRoots = fullRoots[1:]
        AllFiles = AllFiles[1:]

        i = 0
        for path in fullRoots:
            for file in AllFiles[i]:
                with open(path + '/' + file) as f:
                    data = json.load(f)

                diagnFolder = path.split('/')[3]
                pathlib.Path(mainPathTo + diagnFolder).mkdir(parents=True, exist_ok=True)

                datapPeak = data['Signal'][0]
                peaks = biosppy.signals.ecg.christov_segmenter(signal=datapPeak, sampling_rate=200)[0]

                peakCount = 0
                for peak in peaks:
                    dictToWrite = {'diagVec': data['diagVec'], 'Signal': data['Signal'][0][peak - PEAK_MINUS: peak + PEAK_PLUS]}

                    newFileName = file.split('.')[0] + '_' + str(peakCount) + '.' + file.split('.')[1]
                    with open(mainPathTo + diagnFolder + '/' + newFileName, 'w') as outfile:
                        json.dump(dictToWrite, outfile)

                    peakCount+=1

            i += 1
            print(path)
    func(0)


###############################################################################
##      plot ekg in directory
###############################################################################
if plot_ECG_in_directory:
    EKGPath = '../1beat/1beat/Test/others/'
    BIG_JSON = 0
    FULL_EKG = 1
    TARGET_KEY = 'iii'


    if BIG_JSON:

        with open('../ecg_1078/data_1078.json') as f:
            data = json.load(f)

        # for i in data.keys():
        pprint(len(data))

        for key in data.keys():
            signal = data[key]['Leads'][TARGET_KEY]['Signal']

            x = range(len(signal))
            y = signal

            pylab.subplot(1, 1, 1)
            pylab.plot(x, y)
            pylab.title("ekg")

            plt.show()

            input('Enter')

    else:
        tree = os.walk(EKGPath)
        fullRoots = []
        AllFiles = []
        for root, dirs, files in tree:
            AllFiles.append(files)

        AllFiles = AllFiles[0]


        for file in AllFiles:

            with open(EKGPath + file) as f:
                data = json.load(f)


            if FULL_EKG:
                signal = data['Leads'][TARGET_KEY]['Signal']
            else:
                #signal = data['Signal'][0]
                signal = data['Signal']

            x = range(len(signal))
            y = signal


            pylab.subplot(1, 1, 1)
            pylab.plot(x, y)
            pylab.title("ekg")

            plt.show()

            input('Enter')
