# AML_Miniproject1_group12
Active ML and Agency group 12

# Data loading
Loading the data consists of the following structure:

- Each CSV file is one recording consisting of 20 electrodes measuring 50.000 data points
- Each sub folder is one patient, menaing there are 8 patients in total
- Each patient has a recording of both open and closed eyes

- Each electrode in each recording is downsampled from 50.000 to 10.000 data points


Step 1:
- Read all files and load into numpy as either OA or OC
- store patient ID's to avoid data leakage

Step 2:
- Preprocess by downsampling and re-referencing by subtracting the mean across channels

Step 3:
- Convert each recording to a feature vector
- Flatten (20,10.000) matrix to a !D feature vector as this is the format a random forest classifier expects

Step 4:
- Build X and Y
- X: stack all 16 samples in (16, 200.000) matrix
- Y: labels (oa -> 0 and oc -> 1)
- group: patient ID used for patient level splits


