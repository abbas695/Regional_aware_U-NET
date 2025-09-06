import matplotlib.pyplot as plt
from medpy import metric
import numpy as np
import glob
import nibabel as nib
import torch

"""
A group of metrics that can be useful in the process
of comparing the segmentation output to the ground truth
in the medical field 
"""

def assert_shape(test, reference):
    """
    The function takes the test and reference volumes and compares the shapes respectively

    :param test: Prediction
    :param reference: Ground truth
    :return: None
    """
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)

class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):
        """
        Sets the Predicted volume to test
        :param test:
        :return: None
        """
        self.test = test
        self.reset()

    def set_reference(self, reference):
        """
        Sets the Ground truth to reference
        :param reference:
        :return: None
        """
        self.reference = reference
        self.reset()

    def reset(self):
        """
        Resets all the parameters to be used to calculate the metrics
        :return: None
        """
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        """
        Asserts shape and counts all the required variables using the GT and Prediction
        :return: None
        """
        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full

def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    print(tp)
    print(fn)
    print("#########################################")
    return float(tp / (tp + fn))

def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    try:
        return float(tp / (tp + fp))
    except ZeroDivisionError:
        return 0.0

    # return metric.precision(test, reference)

def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting: 
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))

def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    """
    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0
    """
    test, reference = confusion_matrix.test, confusion_matrix.reference
    if (not test.sum()) and (not reference.sum()):
        return 0.0
            # penalize if gt all background, pred has foreground
    elif (not test.sum()) and (reference.sum()):
        return 373.1287
            # penalize if gt has forground, but pred has no prediction
    elif (test.sum()) and (not reference.sum()):
        return 373.1287
    else:
        return metric.hd95(test, reference, voxel_spacing, connectivity)

def hd95(output, target, spacing=None) -> np.ndarray:
    """ output and target should all be boolean tensors"""

    
    B, C = target.shape[:2]
    hd95 = np.zeros((B, C), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            pred, gt = output[b, c], target[b, c]

            # reward if gt all background, pred all background
            if (not gt.sum()) and (not pred.sum()):
                hd95[b, c] = 0.0
            # penalize if gt all background, pred has foreground
            elif (not gt.sum()) and (pred.sum()):
                hd95[b, c] = 1.0
            # penalize if gt has forground, but pred has no prediction
            elif (gt.sum()) and (not pred.sum()):
                hd95[b, c] = 1.0
                #373.1287
            else:
                hd95[b, c] = metric.hd95(pred, gt, voxelspacing=spacing)
    
    return round(hd95.mean(),4)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)

def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    tp = np.logical_and(test == 1, reference == 1).sum()
    fn = np.logical_and(test != 1, reference == 1).sum()
    fp = np.logical_and(test == 1, reference != 1).sum()

    

    if (reference != 1).all():
                # no foreground class
        
        if (test != 1).all() :
            return 1
        else:
            return 0
    ''''
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)
    print("cont_gt: ", np.count_nonzero(reference))
    print("cont_pred: ", np.count_nonzero(test))
    print("#################################################")
    '''''
    nem=float(2* tp )
    dem=float(2 * tp + fp + fn)
    return nem/dem
def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.

    Args:
        y: array-like, shape (n_samples,)
            Class labels to be converted to one-hot representation.
        num_classes: int, optional
            Total number of classes. If not provided, the number of classes
            will be inferred from the input data.

    Returns:
        A binary matrix representation of the input labels.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# pa = "C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2023"
# Actual_list = sorted(glob.glob(pa + '/GT 1417 1441/*'))
# Predicted_list = sorted(glob.glob(pa + '/TEST FOLDER FOR MODELS PREPROCESSING/*'))
# #
# for i in range(len(Actual_list)):
#     GT_seg = np.around(nib.load(Actual_list[i]).get_fdata())
#     seg_pred = np.around(nib.load(Predicted_list[i]).get_fdata())
#     #
#     print((Actual_list[i]))
#     print((Predicted_list[i]))
#     print(np.unique(GT_seg))
#     print(np.unique(seg_pred))
#
#     GT_seg = tf.keras.utils.to_categorical(GT_seg, num_classes=4)
#     # #IF DATA LABELS ARE CONVERTED DO THESE
#     # seg_new = np.zeros_like(seg_pred)
#     # seg_new[seg_pred == 2] = 1
#     # seg_new[seg_pred == 1] = 2
#     # seg_new[seg_pred == 3] = 3
#     # # print(seg_new.shape)
#     #
#     Pred = tf.keras.utils.to_categorical(seg_pred, num_classes=4)
#
#     dice_1 = dice(Pred[:, :, :, 1:], GT_seg[:, :, :, 1:])
#     print(dice_1)
#     # pp = pprint.PrettyPrinter(sort_dicts=True)
#     # pp.pprint(dice)
#     # print(recall(Pred[:, :, :, 1:], GT_seg[:, :, :, 1:]))

#####################
if __name__ == "__main__":
    GT_seg1 = np.around(nib.load(r"G:\files from windows 11\DATA SETS\Brats2023\cycle_gt\BraTS2023_01490_seg.nii.gz").get_fdata())
    seg_pred = np.around(nib.load(r"C:\Users\bedox\Downloads\BraTS2023_01490.nii.gz").get_fdata())

    # print((Actual_list[i]))
    # print((Predicted_list[i]))
    print(np.unique(GT_seg1))
    print(np.unique(seg_pred))
    print(GT_seg1.shape)
    print(seg_pred.shape)
    gt = np.zeros_like(GT_seg1)
    gt[GT_seg1 == 1] = 1
    gt[GT_seg1 == 2] = 2
    gt[GT_seg1 == 4] = 3
    GT_seg = to_categorical(gt, num_classes=4)
    plt.imshow(gt[:, :, 84])
    plt.show()
    # whole_gt = np.logical_or(GT_seg[:, :, :, 1], GT_seg[:, :, :, 2])
    # whole_gt = np.logical_or(whole_gt, GT_seg[:, :, :, 3]).astype(np.uint8)
###################################################
    whole_gt = np.zeros_like(GT_seg1)
    whole_gt[GT_seg1 == 1] = 1
    whole_gt[GT_seg1 == 2] = 1
    whole_gt[GT_seg1 == 4] = 1

    # tumor_core = np.zeros_like(GT_seg1)
    # tumor_core[GT_seg1 == 1] = 0
    # tumor_core[GT_seg1 == 2] = 1
    # tumor_core[GT_seg1 == 3] = 1
    #
    # edema = whole_gt - tumor_core

    # edema = np.zeros_like(GT_seg1)
    # edema[GT_seg1 == 1] = 0
    # edema[GT_seg1 == 2] = 1
    # edema[GT_seg1 == 3] = 0
    #
    # print(np.unique(edema))
    plt.imshow(whole_gt[:, :, 84])
    plt.show()
##############################################################

    # #IF DATA LABELS ARE CONVERTED DO THESE
    seg_new = np.zeros_like(seg_pred)
    seg_new[seg_pred == 1] = 1
    seg_new[seg_pred == 2] = 2
    seg_new[seg_pred == 3] = 3
    # # print(seg_new.shape)
    Pred = to_categorical(seg_new, num_classes=4)
    # print(Pred.shape)
    # seg_pred = seg_pred.T
    plt.imshow(seg_new[:, :, 84])
    plt.show()
    ######################################
    whole_pred = np.zeros_like(seg_pred)
    whole_pred[seg_pred == 1] = 1
    whole_pred[seg_pred == 2] = 1
    whole_pred[seg_pred == 3] = 1
    plt.imshow(whole_pred[:, :, 84])
    plt.show()
    # seg_pred = seg_pred.T
    # edema_seg = seg_pred[:, :, :, 1] - seg_pred[:, :, :, 2]
    # edema_seg[edema_seg == -1] = 0
    # plt.imshow(edema_seg[:, :, 70])
    # plt.show()
    ###########################################
    # whole_pred = np.logical_or(GT_seg[:, :, :, 1], GT_seg[:, :, :, 2])
    # whole_pred = np.logical_or(whole_pred, GT_seg[:, :, :, 3]).astype(np.uint8)

    # dice_1 = dice(GT_seg[:, :, :, 1], Pred[:, :, :, 1])
    # dice_2 = dice(GT_seg[:, :, :, 2], Pred[:, :, :, 2])
    # dice_3 = dice(GT_seg[:, :, :, 3], Pred[:, :, :, 3])
    # dice_0 = dice(GT_seg[:, :, :, 0], Pred[:, :, :, 0])
    # print(dice_0)
    # print(dice_1)
    # print(dice_2)
    # print(dice_3)
    dice1 = hausdorff_distance(whole_pred[:, :, :], whole_gt[:, :, :])
    print(dice1)
    # pp = pprint.PrettyPrinter(sort_dicts=True)
    # pp.pprint(dice)
    # # print(recall(Pred[:, :, :, 1:], GT_seg[:, :, :, 1:]))