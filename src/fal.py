import numpy as np
from utils import model_training, fairness_metrics

class Fal():
    """
        The first fairness-aware AL algorithm that optimizes both group fairness and accuracy. 
        FAL selects the top 𝑚 points with the highest entropy value and then chooses samples that also have the maximum expected reductionin unfairness. 
        A higher 𝑚 favors better fairness.
    """
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, topk, target_fairness, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        
        self.topk = topk
        self.target_fairness = target_fairness
        self.ml_method = ml_method

    def make_query(self, model, num_examples):
        indices = np.arange(len(self.un_label))        
        logits = model.predict_proba(self.un_data)
        
        entropy_arr = np.sum(-logits * np.log(logits), axis=1)
        entropy_arr[np.isnan(entropy_arr)] = 0
        
        selection = np.argsort(entropy_arr)[::-1][:self.topk]
        temp_indices = indices[selection]

        score_arr = []
        for idx in temp_indices:
            score = self.fal_score(logits[idx], idx)
            score_arr.append(score)
            
        max_indices = np.argsort(score_arr)[::-1][:num_examples]
        sel_indices = temp_indices[max_indices]
            
        self.train_data = np.concatenate((self.train_data, self.un_data[sel_indices]), axis=0)
        self.train_label = np.concatenate((self.train_label, self.un_label[sel_indices]), axis=0)
        self.train_z = np.concatenate((self.train_z, self.un_z[sel_indices]), axis=0)
            
        self.un_data = np.delete(self.un_data, sel_indices, axis=0)
        self.un_label = np.delete(self.un_label, sel_indices, axis=0)
        self.un_z = np.delete(self.un_z, sel_indices, axis=0)

        model = model_training(self.train_data, self.train_label, self.val_data, self.val_label, self.ml_method)
        return model
    
    def fal_score(self, un_proba, un_idx):
        train_data = np.concatenate((self.train_data, self.un_data[[un_idx]]), axis=0)
        train_label_0 = np.concatenate((self.train_label, np.zeros_like([un_idx])), axis=0)
        train_label_1 = np.concatenate((self.train_label, np.ones_like([un_idx])), axis=0)
        
        model_0 = model_training(train_data, train_label_0, self.val_data, self.val_label, self.ml_method)
        model_1 = model_training(train_data, train_label_1, self.val_data, self.val_label, self.ml_method)
        
        metric_0, _ = fairness_metrics(model_0, self.val_data, self.val_label, self.val_z, self.target_fairness)
        metric_1, _ = fairness_metrics(model_1, self.val_data, self.val_label, self.val_z, self.target_fairness)

        exp_metric = un_proba[0] * metric_0 + un_proba[1] * metric_1
        return exp_metric