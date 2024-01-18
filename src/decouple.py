import numpy as np
from utils import model_training
    
class Decouple():
    """
        A disagreement-based fairness-aware AL algorithm. 𝐷-𝐹𝐴2𝐿 selects samples for which the decoupled models, 
        trained separately on different sensitive groups, provide different predictions.
    """
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, alpha, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z

        self.alpha = alpha
        self.ml_method = ml_method
        self.num_groups = len(np.unique(self.val_z))

    def make_query(self, num_examples):
        y_hats = []
        for z_value in range(self.num_groups):
            indices = (self.train_z == z_value)            
            model = model_training(self.train_data[indices], self.train_label[indices], self.val_data, self.val_label, self.ml_method)
            y_hat = model.predict_proba(self.un_data)[:,1]
            y_hats.append(y_hat)
        
        diff_indices = np.zeros(len(self.un_label), dtype=bool)
        for i in range(self.num_groups - 1):
            for j in range(i+1, self.num_groups):
                diff_indices += np.abs(y_hats[i] - y_hats[j]) > self.alpha
        
        temp_indices = np.arange(len(self.un_data))[diff_indices]
        if len(temp_indices) < num_examples:
            temp_indices_ = np.random.choice(np.arange(len(self.un_data)), num_examples - len(temp_indices), replace=False)
            sel_indices = np.concatenate((temp_indices, temp_indices_), axis=0)
        else:
            temp_indices = np.arange(len(self.un_data))[diff_indices]
            sel_indices = np.random.choice(temp_indices, num_examples, replace=False)
            
        self.train_data = np.concatenate((self.train_data, self.un_data[sel_indices]), axis=0)
        self.train_label = np.concatenate((self.train_label, self.un_label[sel_indices]), axis=0)
        self.train_z = np.concatenate((self.train_z, self.un_z[sel_indices]), axis=0)
            
        self.un_data = np.delete(self.un_data, sel_indices, axis=0)
        self.un_label = np.delete(self.un_label, sel_indices, axis=0)
        self.un_z = np.delete(self.un_z, sel_indices, axis=0)
        
        model = model_training(self.train_data, self.train_label, self.val_data, self.val_label, self.ml_method)
        return model