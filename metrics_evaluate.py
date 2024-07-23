import numpy as np
import cv2
import torch
from tqdm import tqdm
from losses import BCEDiceLoss
class Eval_MODE():
    def __init__(self):
        self.epsilon = 1e-7
    
    def __call__(self,model,data_loader,device):
        l = self.main(model , data_loader , device)
        return l
    @torch.inference_mode
    def main(self , model , data_loader , device):
        """input ---> model test_data_loader device <<<<"""
        model.eval()
        loss_list = []
        out_list = []
        target_list = []
        criterion = BCEDiceLoss()
        for image , target  in tqdm(data_loader , total=len(data_loader)):
            image , target = image.to(device) , target.to(device)
            with torch.no_grad():
                out = model(image)
                loss = criterion(out , target)
                loss_list.append(loss)
                out_list.append(out)
                target_list.append(target)

        it_range = np.arange(0.1 , 1.01 , 0.01)
        best_th = 0.0
        best_f = 0.0
        best_p = 0.0
        best_r = 0.0
        OIS_RES = self.get_OIS(out_list , target_list)
        for th in it_range:
            th_res , p , r , f= self.get_ODS(out_list, target_list ,th)   
            if f > best_f:
                best_f = f 
                best_th = th_res
                best_p = p
                best_r = r
        print(f"----------> 정밀도평균 : {best_p:.5f} 재현율평균 : {best_r:.5f} OIS : {OIS_RES:.5f}  ODS(F1 score 평균) : {best_f:.5f}")
        return  sum(loss_list) / len(loss_list)


    def get_f1_score(self, outputs , target , th):
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > th).float()
        
        TP = (outputs * target).sum().item()
        FP = (outputs * (1-target)).sum().item()
        FN = ((1-outputs) * target).sum().item()
        
        precision = TP / (TP + FP + self.epsilon)
        recall = TP / (TP + FN + self.epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + self.epsilon)
        
        return f1_score


    def find_best_threshold(self , outputs , target):
        thresholds = np.arange(0, 1.01, 0.01)
        best_f1 = 0.0
        best_thresholds = 0.0
        for i in thresholds:
            f1 = self.get_f1_score(outputs , target , i)
            #print(f"f1 : {f1} threshold : {i}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = i
        return best_thresholds , best_f1
    
    
    def get_OIS(self , out_list : list , target_list : list) :
        f1_list = []
        res = zip(out_list , target_list)
        for i  , (out , target) in enumerate(res):
            output = torch.sigmoid(out)
            best_th , f_ = self.find_best_threshold(output , target)
            print(f"{i}-->Threshold : {best_th:.2f} f1_score : {f_:.4f}")
            f1_list.append(f_)
            
        OIS_RESULT =  sum(f1_list) / len(f1_list)    
        return OIS_RESULT
        
        
    def get_ODS(self , out_list , target_list , th=0.5 , flag=False):
        
        precision_list = []
        recall_list = []
        f1_score_list = []
        res = zip(out_list , target_list)
        for output,target in res:
            output = torch.sigmoid(output)
            
            output = (output > th).float()
            
            TP = (output * target).sum().item()
            FP = (output * (1-target)).sum().item()
            FN = ((1-output) * target).sum().item()
            
            precision = TP / (TP + FP + 1e-7)
            recall = TP / (TP + FN + 1e-7)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

        p = sum(precision_list) / len(precision_list)
        r = sum(recall_list) / len(recall_list)
        f = sum(f1_score_list) / len(f1_score_list)
        return th , p , r , f


