import argparse
import torch
import torch.nn as nn
from network.Math_Module import P, Q
from network.decom import Decom
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
from utils import *


def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class Inference(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # loading decomposition model 
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.Decom_model_low_path)
        # loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L= load_unfolding(self.opts.unfolding_model_path)
        # loading adjustment model
        self.adjust_model = load_adjustment(self.opts.adjust_model_path)
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        print(self.model_Decom_low)
        print(self.model_R)
        print(self.model_L)
        print(self.adjust_model)
        #time.sleep(8)
        
    def get_ratio(self, high_l, low_l):
        ratio = (low_l / (high_l + 0.0001)).mean()
        low_ratio = torch.ones(high_l.shape).cuda() * (1/(ratio+0.0001))
        return low_ratio

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0: # initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else: # update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def lllumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape) * self.opts.ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start = time.time()  
            R, L = self.unfolding(input_low_img)
            High_L = self.lllumination_adjust(L, self.opts.ratio)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def run(self, low_img_path):
        file_name = os.path.basename(self.opts.img_path)
        name = file_name.split('.')[0]
        low_img = self.transform(Image.open(low_img_path)).unsqueeze(0)
        enhance, p_time = self.forward(input_low_img=low_img, input_high_img=None)
        save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_%f_URetinexNet"%(name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)  
        print("================================= time for %s: %f============================"%(file_name, p_time))



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    # specify your data path here!
    parser.add_argument('--img_path', type=str, default="./demo'/input/img.png")
    parser.add_argument('--output', type=str, default="./demo/output")
    # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure 
    parser.add_argument('--ratio', type=float, default=3)
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="./ckpt/decom-L_supervised-4layers")
    parser.add_argument('--unfolding_model_path', type=str, default="./ckpt/HalfDnCNNSER--Illumination_AloneL--A_t_3_RL2-RSSIM-RVGG-concatL-Ltv--1--48___noBN_gamma0.1_lamda0.5_train_from_scratch_offset0.05_Fnorm_Rtv0.0_Ltv20.0--NEW---one_step_model.pthep2000")
    parser.add_argument('--adjust_model_path', type=str, default="./ckpt/t=3ep2000")
    
    opts = parser.parse_args()
    for k, v in vars(opts).items():
        print(k, v)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    model = Inference(opts).cuda()
    model.run()
