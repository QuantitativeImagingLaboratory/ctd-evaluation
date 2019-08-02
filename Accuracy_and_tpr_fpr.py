import csv
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from FilesAccumulator import FilesAccumulator
import matplotlib.pyplot as plt

def loadData(filename, gt = False):
    data = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:

            row_split = row
            if gt:
                data += [[int(row_split[0]), int(row_split[1]), float(row_split[2]), float(row_split[3]), int(row_split[4])]]
            else:

                data += [[int(row_split[0]), int(row_split[1])]]

    index_by_frame = [[]]*(len(data))

    for k in data:
        index_by_frame[k[0]-1] = k[1:]

    return index_by_frame


gt_folder = "Ground_truth"
cam = "cam_a"

gt_cam_folder = os.path.join(gt_folder, cam)

results_folder = "Results"
scenario = "exp_1"

scenario_folder = os.path.join(results_folder,scenario)
this_results_folder = os.path.join(scenario_folder,cam)

files_acum = FilesAccumulator(this_results_folder)
all_files_results = files_acum.find([".csv"])

models = set()
days = set()
for file in all_files_results:
    models.add(file.rsplit("\\",1)[1].split("_")[0])
    days.add(file.rsplit("\\",1)[1].split("_")[1])

TP = 0
FP = 0
TN = 0
FN = 0


for model in models:

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for day in days:
        # hfar_plot = []
        gt_file = os.path.join(gt_cam_folder, day)
        gt_data = loadData(gt_file, gt=True)

        results_file = os.path.join(this_results_folder,model+"_"+day)
        results_data = loadData(results_file)

        y_true = [gt_data[k][0] for k in range(len(results_data))]
        y_prediction = [results_data[k][0] for k in range(len(results_data))]



        # fp = [0] * int(len(y_true)/24)
        for k in range(len(y_true)):

            # if y_prediction[k] > 0 or y_true[k] == 0:
            # if y_true[k] == 2 or y_true[k] == 0:
            #if y_true[k] == 1 or y_true[k] == 2 or y_true[k] == 0:

            if True:
                if y_true[k] == 0 and y_prediction[k] == 0:
                    TN += 1

                elif y_true[k] == 0 and y_prediction[k] > 0:
                    # fp[k] = 1
                    FP += 1

                elif y_true[k] > 0 and y_prediction[k] > 0:
                    TP += 1

                elif y_true[k] > 0 and y_prediction[k] == 0:
                    FN += 1

            # if k % int(len(y_true)/24) == 0:
            #     hfar_plot += [np.sum(fp)]
            #     fp = [0] * len(y_true)


        # plt.plot(hfar_plot)
        # plt.title(model+day)
        # plt.show()
    acc = (TP + TN) / (TP + FP + TN + FN)
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    print("%s, TP: %d, FP: %d, TN: %d, FN: %d, Accuracy: %f, TPR: %f, FPR: %f, hfar:%f" % (model, TP, FP, TN, FN, acc, tpr, fpr, FP/(4*24)))




        # cnf_matrix = confusion_matrix(y_true, y_prediction)
        # # print(cnf_matrix)
        #
        # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        # TP = np.diag(cnf_matrix)
        # TN = cnf_matrix.sum() - (FP + FN + TP)
        #
        # FP = FP.astype(float)
        # FN = FN.astype(float)
        # TP = TP.astype(float)
        # TN = TN.astype(float)
        #
        # # Sensitivity, hit rate, recall, or true positive rate
        # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # Precision or positive predictive value
        # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)
        # # Fall out or false positive rate
        # FPR = FP / (FP + TN)
        # # False negative rate
        # FNR = FN / (TP + FN)
        # # False discovery rate
        # FDR = FP / (TP + FP)
        # # Overall accuracy
        # ACC = (TP + TN) / (TP + FP + FN + TN)

        # print(model, day, TPR)




