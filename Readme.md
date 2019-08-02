# UHCTD-Evaluation

### University of Houston Camera Tampering Detection Evaluation
This code can be used to evaluate the predictions from [ctd-kit](QuantitativeImagingLaboratory/ctd-devkit).
The code can be used to perform evaluaiton under a two class and a four class evaluation. 

To evaluate under a two class scenario, we assume

    +positive class: covered, defocussed, moved
    -negative class: normal 
 
To evaluate under a four class scenario, we assume

|            | class 1 | class 2 | class 3 | class 4 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| +ve class  | normal | covered  | defocussed | moved |
| -ve class  | covered, defocussed, moved | normal, defocussed, moved | normal, covered, moved | normal, defocussed, moved |
 
To reproduce the results from the **UHCTD: A Comprehensive Dataset for Camera Tampering Detection**
The goundtruth and prediciton files are available [here](google.com)
    
**Requirements:** The code has been tested in a windows 10 system.
- Sklearn
- numpy
    
Copy the groundtruth annotation file into the Ground_truth folder. The folder structure is as follows
```
Ground_truth
    |--cam_a
        |--Day 3.csv
        |--Day 4.csv
        ..
    |--cam_b
        |--Day 3.csv
        |--Day 4.csv
        ..
```

Copy the prediction file to Results folder. The folder structure is as follows
```
Results
    |--cam_a
        |--exp_1
            |--alexnet_Day 3.csv
            |--alexnet_Day 4.csv
            ..
        ..
    |--cam_b
        |--exp_1
            |--alexnet_Day 3.csv
            |--alexnet_Day 4.csv
            ..
        ..
```

For two class evaluation run the following command
```
python  Twoclass_evaluation.py
            --exp exp_1
            --cam cam_a            
```

For four class evaluation run the following command
```
python  Fourclass_evaluation.py
            --exp exp_1
            --cam cam_a            
```