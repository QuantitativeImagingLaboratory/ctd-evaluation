import csv
import os
import argparse
from FilesAccumulator import FilesAccumulator

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


def main():
    parser = argparse.ArgumentParser(description='CTD evaluation')
    parser.add_argument('--gt_folder', default='Ground_truth', type=str, help='Specify ground truth folder')
    parser.add_argument('--results_folder', default='Results', type=str, help='Specify results folder')
    parser.add_argument('--cam', default='cam_a', type=str, help='Specify camera (cam_a or cam_b)')
    parser.add_argument('--exp', default='exp_1', type=str, help='Specify experiment (exp_1, exp_3)')
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

            for k in range(len(y_true)):

                if True:
                    if y_true[k] == 0 and y_prediction[k] == 0:
                        TN += 1

                    elif y_true[k] == 0 and y_prediction[k] > 0:
                        FP += 1

                    elif y_true[k] > 0 and y_prediction[k] > 0:
                        TP += 1

                    elif y_true[k] > 0 and y_prediction[k] == 0:
                        FN += 1

        acc = (TP + TN) / (TP + FP + TN + FN)
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        print("%s, TP: %d, FP: %d, TN: %d, FN: %d, Accuracy: %f, TPR: %f, FPR: %f, hfar:%f" % (model, TP, FP, TN, FN, acc, tpr, fpr, FP/(4*24)))


if __name__ == '__main__':
    main()