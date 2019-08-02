# UHCTD-Evaluation

### University of Houston Camera Tampering Detection Evaluation
This code can be used to evaluate the predictions from [ctd-kit](QuantitativeImagingLaboratory/ctd-devkit).
The code can be used to perform evaluaiton under a two class and a four class evaluation. 

To evaluate under a two class scenario, we assume

    +positive class: covered, defocussed, moved
    -negative class: normal 
 
To evaluate under a four class scenario, we assume

|            | Second Header | Second Header | Second Header | Second Header |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| +ve class  | normal | covered  | defocussed | moved |
| -ve class  | covered, defocussed, moved | normal, defocussed, moved | normal, covered, moved | normal, defocussed, moved |
 
To reproduce the results from the **UHCTD: A Comprehensive Dataset for Camera Tampering Detection**
