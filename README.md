# Regional-Aware Clinically Guided Multi-Decoder Pathway Towards Reliable Brain Glioma Segmentation
In this paper, we introduce a novel deep learning framework designed to exploit the clinical utility of MRI sequences, featuring: (1) a multi-decoder architecture that independently segments the non-overlapping tumor subregions, (2) a clinically guided sequence assignment strategy that aligns specific MRI modalities with their diagnostic strengths, and (3) a squeeze-and-excitation (SE) attention mechanism to enhance feature focus and recalibration. More specifically, the clinically guided sequence assignment not only aligns MRI modalities with their diagnostic relevance but also incorporates essential clinical features into dedicated decoders, ensuring each tumor subregion is segmented with modality-specific contextual awareness. 
## Model architechture
<img src="https://github.com/user-attachments/assets/51156cb2-a7a1-414d-85e9-35b7db7dbb4d">

## All the Datasets Used
<img src="https://github.com/user-attachments/assets/355249af-3da2-43ff-9029-494a6fe0d89d">

## Results
### BraTS 2021 & BraTS Africa
<img src='https://github.com/user-attachments/assets/e329fc70-2fb8-4df5-9130-e07b3d7ca026'>

### BraTS 2024 & MRBrainS18
<img src='https://github.com/user-attachments/assets/38b9cbcf-397a-4607-8ba5-ae5920a994b0'>


## Prerequisites
<ul>
  <li>Download the dataset <a href='https://www.med.upenn.edu/cbica/brats2021/#Data2'>from this link</a></li>
  <li>Required folder structure</li>
  
  ```
  /data 
 │
 ├───BraTS2021_train
 │      ├──BraTS2021_00000 
 │      │      └──BraTS2021_00000_flair.nii.gz
 │      │      └──BraTS2021_00000_t1.nii.gz
 │      │      └──BraTS2021_00000_t1ce.nii.gz
 │      │      └──BraTS2021_00000_t2.nii.gz
 │      │      └──BraTS2021_00000_seg.nii.gz
 │      ├──BraTS2021_00002
 │      │      └──BraTS2021_00002_flair.nii.gz
 │      ...    └──...
 │
 └────BraTS2021_val
        ├──BraTS2021_00001 
        │      └──BraTS2021_00001_flair.nii.gz
        │      └──BraTS2021_00001_t1.nii.gz
        │      └──BraTS2021_00001_t1ce.nii.gz
        │      └──BraTS2021_00001_t2.nii.gz
        ├──BraTS2021_00002
        │      └──BraTS2021_00002_flair.nii.gz
        ...    └──...
  ```

  <li>
    <h3>Creare a new environment and in the terminal run the following command to install the required packages from the requirements.txt</h3>

  ``` 
  pip install -r requirements.txt
  ```
  </li>
  
</ul>

## Usage


to run the model follow the instructions from this   <a href='https://github.com/abbas695/Regional_aware_U-NET/blob/main/notebook/regional_aware.ipynb'>notebook<a/>.

## License
This project is licensed under the   Apache License Version 2.0 - see the <a href='https://github.com/abbas695/Regional_aware_U-NET/blob/main/LICENSE'>LICENSE.txt</a> file for details
