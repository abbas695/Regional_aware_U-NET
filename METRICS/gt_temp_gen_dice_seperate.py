import nibabel as nib
import matplotlib.pyplot as plt

import numpy as np
import csv
import time
import glob
import Metrics
import pandas as pd
plt.style.use('dark_background')
import matplotlib as mpl
# mpl.rcParams[
#         'animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'


start = time.time()

# Absolute path to data folder
pa = r'G:\files from windows 11\DATA SETS\Brats2021'
Predicted_list_t1 = sorted(glob.glob('results/tuned_mrbrains/final_preds_please/*'))
Actual_list = sorted(glob.glob('data_brain_mri/BraTS2021_train/labels/*'))

print(len(Predicted_list_t1))
Case_list = []
Dice_list_whole = []
Dice_list_TC = []
Dice_list_ET = []
Dice_list_Edema = []
Dice_list_necrosis = []

Dice_list_agg = []
Dice_list_whole_tumor = []
haus = []
recall = []
recall_whole_list = []
precision = []
precision_whole_list = []
specificity_list = []
sensitivity_list = []
haus_whole_list = []
haus_tc_list = []
haus_en_list = []
haus_edema_list = []
haus_necrosis_list = []
for i in range(len(Actual_list)):
    # Dont use files named with an underscore
    case_num = Predicted_list_t1[i].split("\\")[-1].split(".")[0].split("_")[-1]
    id_val = "Case " + case_num
    Case_list.append(id_val)
    print(id_val, "    ", i)

    GT_seg = np.around(nib.load(Actual_list[i]).get_fdata())
    seg_pred = np.around(nib.load(Predicted_list_t1[i]).get_fdata())

    # seg_pred = seg_pred.T
    # if GT_seg.shape != seg_pred[:, :, :, 0].shape:
    #     print(id_val, "    ", i)


    # THESE LINES ARE FOR WHEN THE DATA IS NOT BRATS convention
    gt = np.zeros_like(GT_seg)
    gt[GT_seg == 1] = 1
    gt[GT_seg == 2] = 2
    gt[GT_seg == 4] = 3
    GT_seg1 = Metrics.to_categorical(gt, num_classes=4)
    
###########################################################################
    """
    EDEMA_gt = np.zeros_like(GT_seg)
    EDEMA_gt[GT_seg == 1] = 0
    EDEMA_gt[GT_seg == 2] = 1
    EDEMA_gt[GT_seg == 3] = 0

    necrosis_gt = np.zeros_like(GT_seg)
    necrosis_gt[GT_seg == 1] = 1
    necrosis_gt[GT_seg == 2] = 0
    necrosis_gt[GT_seg == 3] = 0

    enhancing_tumor_gt = np.zeros_like(GT_seg)
    enhancing_tumor_gt[GT_seg == 1] = 0
    enhancing_tumor_gt[GT_seg == 2] = 0
    enhancing_tumor_gt[GT_seg == 3] = 1

    whole_gt = np.zeros_like(GT_seg)
    whole_gt[GT_seg == 1] = 1
    whole_gt[GT_seg == 2] = 1
    whole_gt[GT_seg == 3] = 1

    tumor_core_gt = np.zeros_like(GT_seg)
    tumor_core_gt[GT_seg == 1] = 1
    tumor_core_gt[GT_seg == 2] = 0
    tumor_core_gt[GT_seg == 3] = 1
    """
###################################################################################
    

###########################################################################
    
    grey_gt = np.zeros_like(GT_seg)
    grey_gt[GT_seg == 1] = 0
    grey_gt[GT_seg == 2] = 1
    grey_gt[GT_seg == 3] = 0

    white_gt = np.zeros_like(GT_seg)
    white_gt[GT_seg == 1] = 1
    white_gt[GT_seg == 2] = 0
    white_gt[GT_seg == 3] = 0

    csf_gt = np.zeros_like(GT_seg)
    csf_gt[GT_seg == 1] = 0
    csf_gt[GT_seg == 2] = 0
    csf_gt[GT_seg == 3] = 1

  

###################################################################################
    # Pred[Pred == 4] = 3
    # will remove this as of the new model

    seg_new_t1 = np.zeros_like(seg_pred)
    seg_new_t1[seg_pred == 1] = 1
    seg_new_t1[seg_pred == 2] = 2
    seg_new_t1[seg_pred == 3] = 3
    Pred = Metrics.to_categorical(seg_new_t1, num_classes=4)

########################################################################
    """
    print(seg_pred.shape)
    print(GT_seg.shape)
  

    en_pred = np.zeros_like(seg_pred[:, :, :])
    en_pred[seg_pred[:, :, :] == 1] = 0
    en_pred[seg_pred[:, :, :] == 2] = 0
    en_pred[seg_pred[:, :, :] == 3] = 1

    whole_tumor_pred = np.zeros_like(seg_pred[:, :, :])
    whole_tumor_pred[seg_pred[:, :, :] == 1] = 1
    whole_tumor_pred[seg_pred[:, :, :] == 2] = 1
    whole_tumor_pred[seg_pred[:, :, :] == 3] = 1

    tumor_core_pred = np.zeros_like(seg_pred[:, :, :])
    tumor_core_pred[seg_pred[:, :, :] == 2] = 0
    tumor_core_pred[seg_pred[:, :, :] == 1] = 1
    tumor_core_pred[seg_pred[:, :, :] == 3] = 1
    """
#####################################################################

########################################################################
    
    print(seg_pred.shape)

    white_pred = seg_pred[1,:, :, :]
  

    grey_pred = seg_pred[0,:, :, :]
   

    csf_pred = seg_pred[2,:, :, :]

    
#####################################################################
    # seg_pred = seg_pred.T
    dice_ENHANCING = Metrics.dice(grey_pred, grey_gt)
    dice_whole = Metrics.dice(white_pred, white_gt)
    dice_core = Metrics.dice(csf_pred, csf_gt)
    hausdorff_whole = Metrics.hausdorff_distance(white_pred, white_gt)
    hausdorff_core = Metrics.hausdorff_distance(csf_pred, csf_gt)
    hausdorff_en = Metrics.hausdorff_distance(grey_pred, grey_gt)
    print((dice_whole+dice_core+dice_ENHANCING)/3)

    #print(hous_whole)
    # dice_nec = Metrics.precision(whole_pred[:                                                                                                                                                                             , :, :], whole_gt[:, :, :])
    # dice_ed = Metrics.precision(tc_pred[:, :, :], tumor_core[:, :, :])
    # dice_et = Metrics.precision(en_pred[:, :, :], enhancing_tumor[:, :, :])

    # dice_nec = Metrics.dice(Pred[:, :, :, 1], GT_seg1[:, :, :, 1])
    # dice_ed = Metrics.dice(Pred[:, :, :, 2], GT_seg1[:, :, :, 2])
    # dice_et = Metrics.dice(Pred[:, :, :, 3], GT_seg1[:, :, :, 3])

    # hausdorff = Metrics.hausdorff_distance(Pred[:, :, :, 1:], GT_seg1[:, :, :, 1:])
    # agg = Metrics.dice(Pred[:, :, :, 1:], GT_seg1[:, :, :, 1:])
    # specificity = Metrics.specificity(Pred[:, :, :, 1:], GT_seg1[:, :, :, 1:])
    # sensitivity = Metrics.sensitivity(Pred[:, :, :, 1:], GT_seg1[:, :, :, 1:])
    # dice = Metrics.dice(Pred[:, :, :, 1:], GT_seg1[:, :, :, 1:])
    # seg_pred = seg_pred.T
    # edema_seg = seg_pred[:, :, :, 1] - seg_pred[:, :, :, 2]
    # edema_seg[edema_seg == -1] = 0
    # dice = Metrics.dice(seg_pred[:, :, :, 2], whole_gt[:, :, :])
##########################################################################################################################
    # recall_ed = Metrics.recall(Pred[:, :, :, 2], GT_seg1[:, :, :, 2])
    # recall_whole = Metrics.recall(whole_pred[:, :, :], whole_gt[:, :, :])
    #
    # precision_ed = Metrics.precision(Pred[:, :, :, 2], GT_seg1[:, :, :, 2])
    # precision_whole = Metrics.precision(whole_pred[:, :, :], whole_gt[:, :, :])
##########################################################################################################################
    # Dice_list_nec.append(dice_nec)
    # Dice_list_ed.append(dice_ed)
    # Dice_list_et.append(dice_et)
    Dice_list_Edema.append(dice_whole)
    Dice_list_necrosis.append(dice_core)
    Dice_list_ET.append(dice_ENHANCING)
    #Dice_list_Edema.append(dice_EDEMA)
    #Dice_list_necrosis.append(dice_NECROSIS)
    haus_en_list.append(hausdorff_en)
    haus_edema_list.append(hausdorff_whole)
    haus_necrosis_list.append(hausdorff_core)
    #haus.append(hausdorff)
    # recall.append(agg)
    # specificity_list.append(specificity)
    # sensitivity_list.append(sensitivity)

    # Dice_list_agg.append(dice)
    # print(dice_ed)
    # Dice_list_whole_tumor.append(dice)
##########################################################################################################################
    # recall.append(recall_ed)
    # recall_whole_list.append(recall_whole)
    # precision.append(precision_ed)
    # precision_whole_list.append(precision_whole)
##########################################################################################################################
# zipped = list(zip(Case_list, Dice_list_nec, Dice_list_ed, Dice_list_et, haus_whole_list, haus_tc_list, haus_en_list))
#,'whole_hd95', 'tc_hd95', 'et_hd95'
zipped = list(zip(Case_list,Dice_list_Edema,Dice_list_necrosis,Dice_list_ET,haus_edema_list,haus_necrosis_list,haus_en_list))
# df = pd.DataFrame(zipped, columns=['Patient', 'precision whole', 'precision tumor core', 'precision ET'
#                                    , 'sensitivity whole', 'sensitivity tumor core', 'sensitivity ET'])
df = pd.DataFrame(zipped, columns=['Patient','dice_white', 'dice_csf', 'dice_grey','hausdorff_white', 'hausdorff_csf', 'hausdorff_grey'])
print(df)
df.to_csv(r'results/tuned_mrbrains/tuned_mrbrains _dsc_hd95.csv')

end = time.time()
print(end - start)  
