# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hiddenlayer as hl
import numpy as np
import pytorch_lightning as pl
import torch
from monai.visualize import GradCAM
import statistics
import torch.nn as nn
import matplotlib.pyplot as plt
from apex.optimizers import FusedAdam, FusedSGD
from data_loading.data_module import get_data_path, get_test_fnames
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet,SegResNetVAE,AttentionUnet,BasicUNetPlusPlus,UNETR,ViTAutoEnc,SwinUNETR
from nnunet.regional_aware_model import UNet3D
from nnunet.loss import Loss, LossBraTS
from nnunet.metrics import Dice
from pytorch_lightning.utilities import rank_zero_only
from scipy.special import expit, softmax
from skimage.transform import resize
from utils.logger import DLLogger
from utils.utils import get_config_file, print0
import torch
from torch.optim.lr_scheduler import _LRScheduler
import random
import math
import warnings

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class WarmStartReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 warm_start: float,
                 warm_stop: float,
                 warm_patience: int = 0,
                 warm_duration: int = 25,
                 warm_type: str = "linear",
                 mode: str = "min",
                 patience: int = 10,
                 cooldown=0,
                 factor=0.1,
                 threshold=1e-4,
                 threshold_mode='rel',
                 min_lr=0,
                 eps=1e-8,
                 verbose=False):
        """
        Workaround class as SequentialLR with ReduceLROnPlateau is not working in pytorch lightning currently.
        Otherwise simply use WarmStart class together with any of the other pytorch schedulers.

        See Also
        https://github.com/PyTorchLightning/pytorch-lightning/issues/10759
        """
        assert warm_type in ("linear", "smooth")
        assert warm_duration > 0
        assert warm_patience >= 0

        self.warm_start = warm_start
        self.warm_stop = warm_stop
        self.warm_patience = warm_patience
        self.warm_duration = warm_duration
        self.warm_type = warm_type
        self.warm_ended = False
        self._last_lr = warm_start

        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose
        )

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check if out of warm-up patience period and if warm-up should end
        if self.last_epoch > self.warm_patience and not self.warm_ended:
            self._warm_lr(self.last_epoch)

        # Check if out of warm-up phase
        if self.last_epoch > self.warm_patience + self.warm_duration:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                # Indicate to warm up that LRReduce should happen; prevent LR override
                if self.verbose and not self.warm_ended:
                    print(f"Ending warm-up phase after {epoch} epochs. "
                          f"Switching over to ReduceLROnPlateau")
                self.warm_ended = True

                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _warm_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            slope = (self.warm_stop - self.warm_start)
            x = (epoch - self.warm_patience) / self.warm_duration
            lower_bound = min(self.warm_start, self.warm_stop)
            upper_bound = max(self.warm_start, self.warm_stop)
            if self.warm_type == "linear":
                new_lr = slope * x + self.warm_start
            else:
                new_lr = slope * math.tanh(x) + self.warm_start
            param_group['lr'] = np.clip(new_lr, lower_bound, upper_bound)
            if self.verbose and not np.isclose(old_lr, new_lr):
                print('Epoch {:5d}: warming-up learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
class NNUnet(pl.LightningModule):
    def __init__(self, args, triton=False, data_dir=None):
        super(NNUnet, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.triton = triton
        if data_dir is not None:
            self.args.data = data_dir
        self.build_nnunet()
        self.best_mean, self.best_epoch, self.test_idx = (0,) * 3
        self.start_benchmark = 0
        self.train_loss = []
        self.test_imgs = []
        if not self.triton:
            self.learning_rate = args.learning_rate
            loss = LossBraTS if self.args.brats else Loss
            self.loss = loss(self.args.focal)
            if self.args.dim == 2:
                self.tta_flips = [[2], [3], [2, 3]]
            else:
                self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
            self.dice = Dice(self.n_class, self.args.brats)
            if self.args.exec_mode in ["train", "evaluate"] and not self.args.benchmark:
                self.dllogger = DLLogger(args.results, args.logname)

    def forward(self, img):
        return torch.argmax(self.model(img), 1)

    def _forward(self, img):
        if self.args.benchmark:
            if self.args.dim == 2 and self.args.data2d_dim == 3:
                img = layout_2d(img, None)
            return self.model(img)
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def compute_loss(self, preds, label):
        
        if self.args.regional_aware:
            loss = self.loss(preds[0], label,self.current_epoch)
            weights = 0.0
            for i, pred in enumerate(preds[1:]):
                upsampled_pred = nn.functional.interpolate(pred, label.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(upsampled_pred, label,self.current_epoch)
                weights += 0.5**(i+1)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return loss
  
        if self.args.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(preds.shape[1]):
                loss += self.loss(preds[:, i], label,self.current_epoch) * 0.5**i
                weights += 0.5**i
            return loss / weights
        return self.loss(preds, label,self.current_epoch)

    def training_step(self, batch, batch_idx):
        img, lbl = self.get_train_data(batch)
        img, lbl = self.convert_data(img, lbl)
        pred = self.model(img)
        loss = self.compute_loss(pred, lbl)
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        
        if self.current_epoch < self.args.skip_first_n_eval:
            return None
        
        img, lbl = batch["image"], batch["label"]
        
        img, lbl = self.convert_data(img, lbl)
     
        pred = self._forward(img)
        loss = self.loss(pred, lbl,self.current_epoch)
        if self.args.invert_resampled_y:
            meta, lbl = batch["meta"][0].cpu().detach().numpy(), batch["orig_lbl"]
            pred = nn.functional.interpolate(pred, size=tuple(meta[3]), mode="trilinear", align_corners=True)
        self.dice.update(pred, lbl[:, 0], loss)

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        img = self.convert_ncdhw_to_ndhwc(img)
        if self.args.benchmark:
            pred = self._forward(img)
            return

        pred = self._forward(img).squeeze(0).cpu().detach().numpy()
        if self.args.save_preds:
            meta = batch["meta"][0].cpu().detach().numpy()
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]
            n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]
            if not all(cropped_shape == pred.shape[1:]):
                resized_pred = np.zeros((n_class, *cropped_shape))
                for i in range(n_class):
                    resized_pred[i] = resize(
                        pred[i], cropped_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                    )
                pred = resized_pred
            final_pred = np.zeros((n_class, *original_shape))
            final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            if self.args.brats:
                final_pred = expit(final_pred)
            else:
                final_pred = softmax(final_pred, axis=0)

            self.save_mask(final_pred)

    def get_unet_params(self):
        config = get_config_file(self.args)
        patch_size, spacings = config["patch_size"], config["spacings"]
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [
                2 if ratio <= 2 and size >= 2 * self.args.min_fmap else 1 for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == self.args.depth:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return config["in_channels"], config["n_class"], kernels, strides, patch_size

    def convert_ncdhw_to_ndhwc(self, tensor):
        if self.args.layout == "NCDHW":
            return tensor
        strides = tensor.stride()
        shape = tensor.shape
        tensor = torch.as_strided(
            tensor, (shape[0], shape[-1], *shape[1:-1]), (strides[0], strides[-1], *strides[1:-1])
        )
        return tensor

    def convert_data(self, img, lbl):
        img, lbl = self.convert_ncdhw_to_ndhwc(img), self.convert_ncdhw_to_ndhwc(lbl)
        return img, lbl

    def build_nnunet(self):
        self.in_channels, out_channels, kernels, strides, self.patch_size = self.get_unet_params()
        self.n_class = out_channels - 1
        if self.args.brats:
            out_channels = 3

        if self.args.regional_aware:
            self.model = UNet3D(kernels, strides)
        
        else:
            
            self.model = DynUNet(
                self.args.dim,
                self.in_channels,
                out_channels,
                kernels,
                strides,
                strides[1:],
                filters=self.args.filters,
                norm_name=(self.args.norm.upper(), {"affine": True}),
                act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
                deep_supervision=self.args.deep_supervision,
                deep_supr_num=self.args.deep_supr_num,
                res_block=self.args.res_block,
                trans_bias=True,
            )
            


        """
        g = hl.build_graph(self.model, torch.zeros([1, 5, 128, 128, 128]))
        g.save("Architecture/network_architecture.pdf")
        del g
        
        for name, _ in self.model.named_modules(): print(name)
        self.cam = GradCAM(nn_module=self.model, target_layers="upsamples.5.conv_block.conv2.conv")
        numpy_array = np.load('/home/abbas/multible_decoder_trials/nnUNet/paper/gradcam/input0.npy')
        tensor = torch.from_numpy(numpy_array)
        results=self.cam(tensor,class_idx=2)
        np.save(f"/home/abbas/multible_decoder_trials/nnUNet/paper/gradcam/gradcam_outputs/output0",results.cpu().numpy())
        print(results.shape)
        if self.args.layout == "NDHWC" and self.args.dim == 3:
            self.model.to(memory_format=torch.channels_last_3d)
        """
        #print0(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds = self.model(image)
        preds = torch.transpose(preds, 0, 1).unsqueeze(0)
        return preds

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    def round(self, tensor):
        return round(torch.mean(tensor).item(), 2)

    def on_validation_epoch_end(self):
       
        metrics = {}
        if self.current_epoch < self.args.skip_first_n_eval:
            self.log("dice", 0.0, sync_dist=False)
            self.dice.reset()
            metrics["Train Loss"] = (
            0 if len(self.train_loss) == 0 else round(sum(self.train_loss) / len(self.train_loss), 10)
        )
            #metrics["learning rate"] = self.lr_schedulers().get_last_lr()
            self.log("train_loss", metrics["Train Loss"] ,sync_dist=False)
            
            self.dllogger.log_metrics(step=self.current_epoch, metrics=metrics)
            self.dllogger.flush()
            return None
        dice, loss = self.dice.compute()
        self.dice.reset()
        # Update metrics
        dice_mean = torch.mean(dice)
        if dice_mean >= self.best_mean:
            self.best_mean = dice_mean
            self.best_mean_dice = dice
            self.best_epoch = self.current_epoch
        metrics["Dice"] = self.round(dice)
        metrics["Val Loss"] = self.round(loss)
        metrics["Max Dice"] = self.round(self.best_mean_dice)
        metrics["Best epoch"] = self.best_epoch
        metrics["Train Loss"] = (
            0 if len(self.train_loss) == 0 else round(sum(self.train_loss) / len(self.train_loss), 5)
        )
        #metrics["learning rate"] = self.lr_schedulers().get_last_lr()
        if self.n_class > 1:
            metrics.update({f"D{i+1}": self.round(m) for i, m in enumerate(dice)})

        self.dllogger.log_metrics(step=self.current_epoch, metrics=metrics)
        self.dllogger.flush()
        self.log("dice", metrics["Dice"],sync_dist=False)
        self.log("val_loss", loss ,sync_dist=False)
        self.log("train_loss", metrics["Train Loss"] ,sync_dist=False)


    def on_test_epoch_end(self):
        if self.args.exec_mode == "evaluate":
            self.eval_dice, _ = self.dice.compute()

    @rank_zero_only
    def on_fit_end(self):
        if not self.args.benchmark:
            metrics = {}
            metrics["dice_score"] = round(self.best_mean.item(), 2)
            metrics["train_loss"] = round(sum(self.train_loss) / len(self.train_loss), 4)
            metrics["val_loss"] = round(1 - self.best_mean.item() / 100, 4)
            metrics["Epoch"] = self.best_epoch
            self.dllogger.log_metrics(step=(), metrics=metrics)
            self.dllogger.flush()

    def configure_optimizers(self):
        optimizer = {
            "sgd": FusedSGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=True),
            "adam": FusedAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        if self.args.scheduler:
           """
           scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4096, eta_min=8e-5)
           return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}
           """
           scheduler=WarmStartReduceOnPlateau(optimizer, mode='min', factor=0.95, patience=0,
                                                        threshold=0.0001, threshold_mode='abs',
                                                        cooldown=0, min_lr=0, eps=1e-08,warm_duration=1,warm_start=0.0003,warm_stop=0.0003,warm_patience=0)
           return {"optimizer": optimizer, "monitor": "train_loss", "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train_loss",
                    },}
                    
        return {"optimizer": optimizer, "monitor": "val_loss"}
    

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_data_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        return img, lbl
    
    def get_current_epoch(self):
        return self.current_epoch
    

def layout_2d(img, lbl):
    batch_size, depth, channels, height, width = img.shape
    img = torch.reshape(img, (batch_size * depth, channels, height, width))
    if lbl is not None:
        lbl = torch.reshape(lbl, (batch_size * depth, 1, height, width))
        return img, lbl
    return img


def flip(data, axis):
    return torch.flip(data, dims=axis)
