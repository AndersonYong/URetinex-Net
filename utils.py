import torchvision
from torch.nn import init
import numpy as np
import os
import time
import torch
from PIL import Image
import glob

def save_TensorImg(img_tensor, path, nrow=1):
    torchvision.utils.save_image(img_tensor, path, nrow=nrow)

def np_save_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img*255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def define_modelR(opts):
    if opts.R_model == "HalfDnCNNSE":
        from network.restoration import HalfDnCNNSE
        model_R = HalfDnCNNSE(opts)
    return model_R

def define_modelL(opts):
    if opts.L_model == "Illumination_Alone":
        from network.illumination_enhance import Illumination_Alone
        model_L = Illumination_Alone(opts)
    return model_L

def define_modelA(opts):
    if opts.A_model == "naive":
        from network.illumination_adjustment import Adjust_naive
        model_A = Adjust_naive(opts)
    return model_A



def load_initialize(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path)
        model.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # to freeze the params of Decomposition Model
        for param in model.parameters():
            param.requires_grad = False   
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

def load_unfolding(unfolding_model_path):
    if os.path.exists(unfolding_model_path):
        checkpoint = torch.load(unfolding_model_path)
        old_opts = checkpoint["opts"]
        model_R = define_modelR(old_opts)
        model_L = define_modelL(old_opts)
        model_R.load_state_dict(checkpoint['state_dict']['model_R'])
        model_L.load_state_dict(checkpoint['state_dict']['model_L'])
        for param_R in model_R.parameters():
            param_R.requires_grad = False
        for param_L in model_L.parameters():
            param_L.requires_grad = False
        return old_opts, model_R, model_L
    else:
        print("pretrained Unfolding Model does not exist, check ---> %s"%unfolding_model_path)
        exit()

def load_adjustment(adjust_model_path):
    if os.path.exists(adjust_model_path):
        checkpoint_Adjust = torch.load(adjust_model_path)
        model_A = define_modelA(checkpoint_Adjust['opts'])
        model_A.load_state_dict(checkpoint_Adjust['state_dict']['model_A'])
        print(" ===========>  loading pretrained Illumination Adjustment Model from: %s " % adjust_model_path)
        # to freeze the params of Decomposition Model
        for param in model_A.parameters():
            param.requires_grad = False       
        return model_A
    else:
        print("pretrained Adjustment Model does not exist, check ---> %s"%adjust_model_path)
        exit()
    

        

def param_all(model,  net_input):
    import torchsummary
    shape = net_input.shape
    torchsummary.summary(model, (shape[1], shape[2], shape[3]))

def param_self_compute(model):
    parmas = 0
    for p in model.parameters():
        #print(p)
        parmas += p.numel()
    return parmas




