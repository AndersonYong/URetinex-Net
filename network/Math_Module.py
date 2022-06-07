import torch
import torch.nn as nn
from torchvision.transforms import Grayscale


class P(nn.Module):
    """
        to solve min(P) = ||I-PQ||^2 + γ||P-R||^2
        this is a least square problem
        how to solve?
        P* = (gamma*R + I*Q) / (Q*Q + gamma)
    """
    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q))
        
class Q(nn.Module):
    """
        to solve min(Q) = ||I-PQ||^2 + λ||Q-L||^2
        Q* = (lamda*L + I*P) / (P*P + lamda)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, I, P, L, lamda):
        
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]

        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]

        return (IR*PR + IG*PG + IB*PB + lamda*L) / ((PR*PR + PG*PG + PB*PB) + lamda)
        