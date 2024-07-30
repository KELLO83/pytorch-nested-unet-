from archs import *
import torchinfo

model = AttentionResUNet(in_channels=3 , out_channels=1)
torchinfo.summary(model , input_size=(1,3,512,512))