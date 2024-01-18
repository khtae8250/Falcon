import numpy as np

from utils import model_training

class Entropy():
    """
        A standard AL algorithm that selects the most uncertain samples based on entropy
    """
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z      
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        self.ml_method = ml_method

    def make_query(self, model, num_examples):
        indices = np.arange(len(self.un_label))
        target_indices = indices

        logits = model.predict_proba(self.un_data[target_indices])
        entropy_arr = np.sum(-logits * np.log(logits), axis=1)
        entropy_arr[np.isnan(entropy_arr)] = 0
        
        selection = np.argsort(entropy_arr)[::-1][:num_examples]
        temp_indices = indices[target_indices][selection]
        sel_indices = temp_indices
            
        self.train_data = np.concatenate((self.train_data, self.un_data[sel_indices]), axis=0)
        self.train_label = np.concatenate((self.train_label, self.un_label[sel_indices]), axis=0)
        self.train_z = np.concatenate((self.train_z, self.un_z[sel_indices]), axis=0)
            
        self.un_data = np.delete(self.un_data, temp_indices, axis=0)
        self.un_label = np.delete(self.un_label, temp_indices, axis=0)
        self.un_z = np.delete(self.un_z, temp_indices, axis=0)
        
        model = model_training(self.train_data, self.train_label, self.val_data, self.val_label, self.ml_method)
        return model
        
    def update(self, train_data, train_label, train_z, un_data, un_label, un_z):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z