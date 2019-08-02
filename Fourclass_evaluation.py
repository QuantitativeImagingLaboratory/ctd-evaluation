import csv
import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from FilesAccumulator import FilesAccumulator


def loadData(filename, gt = False):
    data = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(row[0].split(','))

            row_split = row
            if gt:
                data += [[int(row_split[0]), int(row_split[1]), float(row_split[2]), float(row_split[3]), int(row_split[4])]]
            else:

                data += [[int(row_split[0]), int(row_split[1])]]

    index_by_frame = [[]]*(len(data))

    for k in data:
        index_by_frame[k[0]-1] = k[1:]

    return index_by_frame

def main():
    parser = argparse.ArgumentParser(description='CTD evaluation')
    parser.add_argument('--gt_folder', default='Ground_truth', type=str, help='Specify ground truth folder')
    parser.add_argument('--results_folder', default='Results', type=str, help='Specify results folder')
    parser.add_argument('--cam', default='cam_a', type=str, help='Specify camera (cam_a or cam_b)')
    parser.add_argument('--exp', default='exp_1', type=str, help='Specify experiment (exp_1, exp_2, exp_3)')
    args = parser.parse_args()

    gt_folder = args.gt_folder
    cam = args.cam

    gt_cam_folder = os.path.join(gt_folder, cam)

    results_folder = args.results_folder
    scenario = args.exp

    scenario_folder = os.path.join(results_folder,scenario)
    this_results_folder = os.path.join(scenario_folder,cam)

    files_acum = FilesAccumulator(this_results_folder)
    all_files_results = files_acum.find([".csv"])

    models = set()

    days = set()
    for file in all_files_results:
        models.add(file.rsplit("\\",1)[1].split("_")[0])
        days.add(file.rsplit("\\",1)[1].split("_")[1])

    models = ("alexnet", "resnet18", "resnet50", "densenet161")


    for model in models:
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for day in days:

            gt_file = os.path.join(gt_cam_folder, day)
            gt_data = loadData(gt_file, gt=True)

            results_file = os.path.join(this_results_folder,model+"_"+day)
            results_data = loadData(results_file)

            y_true = [gt_data[k][0] for k in range(len(results_data))]
            y_prediction = [results_data[k][0] for k in range(len(results_data))]

            cnf_matrix = confusion_matrix(y_true, y_prediction)

            FP1 = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN1 = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP1 = np.diag(cnf_matrix)
            TN1 = cnf_matrix.sum() - (FP1 + FN1 + TP1)

            FP += FP1
            FN += FN1
            TP += TP1
            TN += TN1

        # Reference: https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        # Ignoring normal and moved classes for experiment 2
        if scenario == 'exp_2':
            TPR = TPR * [0,1,1,0]
            FPR = FPR * [0,1,1,0]
            ACC = ACC * [0,1,1,0]

        accum = ["%0.2f & %0.2f & %0.2f"%(a, b, c) for (a, b, c) in zip(TPR, FPR, ACC)]
        print(model, accum)


if __name__ == '__main__':
    main()