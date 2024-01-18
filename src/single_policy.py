import numpy as np
from utils import model_training
        
class Single_policy():
    """
        Baseline method 𝑃𝑜𝑙(𝑥) for policy r=x randomly selects a target group (if there are more than one) for each labeling round 
        and selects a sample whose probability for the desirable label is closest to 1 − x.
    """
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, r, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        self.r = r
        self.ml_method = ml_method

    def make_query(self, model, target_group, num_examples):
        indices = np.arange(len(self.un_label))        
        target_indices = (self.un_z == target_group[1]) 
            
        logits = model.predict_proba(self.un_data[target_indices])
        r_arr = - np.abs(logits[:,target_group[0]] - (1 - self.r))
        
        selection = np.argsort(r_arr)[::-1][:num_examples]
        temp_indices = indices[target_indices][selection]
        sel_indices = temp_indices[self.un_label[temp_indices] == target_group[0]]
            
        self.train_data = np.concatenate((self.train_data, self.un_data[sel_indices]), axis=0)
        self.train_label = np.concatenate((self.train_label, self.un_label[sel_indices]), axis=0)
        self.train_z = np.concatenate((self.train_z, self.un_z[sel_indices]), axis=0)
            
        self.un_data = np.delete(self.un_data, temp_indices, axis=0)
        self.un_label = np.delete(self.un_label, temp_indices, axis=0)
        self.un_z = np.delete(self.un_z, temp_indices, axis=0)
        
        model = model_training(self.train_data, self.train_label, self.val_data, self.val_label, self.ml_method)
        return model