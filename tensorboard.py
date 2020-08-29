import torch
from tensorboardX import SummaryWriter
import etm

PATH = './results/etm_20ng_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(PATH, map_location=torch.device('cpu'))

writer = SummaryWriter()
rho_weights = model.rho.weight

writer.add_embedding(torch.FloatTensor(rho_weights))