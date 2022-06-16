import argparse
from fileinput import filename
from locale import locale_encoding_alias
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
import glob

"""
  As different illumination adjustment ratio will cause
different enhanced results. Certainly you can tune the ratio youself
to get the best results.
  To get better result, we use the illumination of normal light image 
to adaptively generate ratio.
  Noted that KinD and KinD++ also use ratio to guide the illumination adjustment,
for fair comparison, the ratio of their methods also generate by the illumination
of normal light image.
"""

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class Inference(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # loading decomposition model 
        self.model_Decom_low = Decom()
        self.model_Decom_high = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.Decom_model_low_path)
        self.model_Decom_high = load_initialize(self.model_Decom_high, self.opts.Decom_model_high_path)
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
        ratio = torch.ones(L.shape).cuda() * ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img, input_high_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            input_high_img = input_high_img.cuda()
        with torch.no_grad():
            start = time.time()  
            R, L = self.unfolding(input_low_img)
            # the ratio is calculated using the decomposed normal illumination
            _, high_L = self.model_Decom_high(input_high_img)
            ratio = self.get_ratio(high_L, L)
            High_L = self.lllumination_adjust(L, ratio)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def evaluate(self):
        low_files = glob.glob(self.opts.low_dir+"/*.png")
        for file in low_files:
            file_name = os.path.basename(file)
            name = file_name.split('.')[0]
            high_file = os.path.join(self.opts.high_dir, file_name)
            low_img = self.transform(Image.open(file)).unsqueeze(0)
            high_img = self.transform(Image.open(high_file)).unsqueeze(0)
            enhance, p_time = self.forward(low_img, high_img)
            if not os.path.exists(self.opts.output):
                os.makedirs(self.opts.output)
            save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_URetinexNet"%(name)))
            np_save_TensorImg(enhance, save_path)  
            print("================================= time for %s: %f============================"%(file_name, p_time))



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    # specify your data path here!
    parser.add_argument('--low_dir', type=str, default="./test_daat/LOLdataset/eval15/low")
    parser.add_argument('--high_dir', type=str, default="./test_data/LOLdataset/eval15/high")
    parser.add_argument('--output', type=str, default="./demo/output/LOL")
    # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure 
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="./ckpt/init_low.pth")
    parser.add_argument('--Decom_model_high_path', type=str, default="./ckpt/init_high.pth")
    parser.add_argument('--unfolding_model_path', type=str, default="./ckpt/unfolding.pth")
    parser.add_argument('--adjust_model_path', type=str, default="./ckpt/L_adjust.pth")
    parser.add_argument('--gpu_id', type=int, default=0)
    
    opts = parser.parse_args()
    for k, v in vars(opts).items():
        print(k, v)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    model = Inference(opts).cuda()
    model.evaluate()
