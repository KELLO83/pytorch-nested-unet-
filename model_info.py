from archs import NestedUNet
import torchinfo

model = NestedUNet(num_classes=1)
torchinfo.summary(model , input_size=(1,3,512,512))