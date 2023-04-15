import torch
import torch.nn as nn
from network.Math_Module import P, Q
from network.decom import Decom

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
from utils import *

class Inference(nn.Module):
    def __init__(self):
        super().__init__()

        self.Decom_model_low_path = "./ckpt/init_low.pth"
        self.unfolding_model_path = "./ckpt/unfolding.pth"
        self.adjust_model_path = "./ckpt/L_adjust.pth"
        self.ratio = 3
        self.output = "./demo/output"
        self.img_path = "./demo/input/3.png"
        
        # loading decomposition model 
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.Decom_model_low_path)
        # loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L= load_unfolding(self.unfolding_model_path)
        # loading adjustment model
        self.adjust_model = load_adjustment(self.adjust_model_path)
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        ratio = torch.ones(L.shape).to(self.device) * self.ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        input_low_img = input_low_img.to(self.device)
        with torch.no_grad():
            start = time.time()  
            R, L = self.unfolding(input_low_img)
            High_L = self.lllumination_adjust(L, self.ratio)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def runforstreamlit(image):
        
        # Load the image
        img = self.transform(image)
        img = img.unsqueeze(0)
        enhance, p_time = self.forward(input_low_img=img)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        save_path = os.path.join(self.output, "1.png")
        np_save_TensorImg(enhance, save_path)  
        result = Image.open(save_path)
        return result
    
model = Inference()
