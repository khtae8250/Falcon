import numpy as np
from utils import model_training, fairness_metrics
from entropy import Entropy

class Combined():
    """
        A fair active learning framework Falcon that selects samples for the purpose of improving fairness and accuracy.
    """
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, policy_set, lambda_, target_fairness, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        
        self.policy_set = policy_set
        self.lambda_ = lambda_
        self.target_fairness = target_fairness
        self.ml_method = ml_method

        self.falcon = Falcon(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z, self.val_data, self.val_label, self.val_z, self.target_fairness, self.ml_method)
        self.entropy =  Entropy(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z, self.val_data, self.val_label, self.val_z, self.ml_method)

    def make_query(self, prev_model, target_groups, num_examples):
        method_id = np.random.choice([0, 1], size=1, p=[self.lambda_, 1-self.lambda_])[0]
        if method_id == 0:
            self.falcon.update(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z)
            model= self.falcon.make_query(prev_model, num_examples, target_groups, self.policy_set)
            self.update(self.falcon.train_data, self.falcon.train_label, self.falcon.train_z, self.falcon.un_data, self.falcon.un_label, self.falcon.un_z)
        else:
            self.entropy.update(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z)
            model = self.entropy.make_query(prev_model, num_examples)
            self.update(self.entropy.train_data, self.entropy.train_label, self.entropy.train_z, self.entropy.un_data, self.entropy.un_label, self.entropy.un_z)
        
        return model
    
    def update(self, train_data, train_label, train_z, un_data, un_label, un_z):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z

class Single_policy():
    def __init__(self, target_group, r):
        self.target_group = target_group
        self.r = r

    def make_query(self, model, un_data, un_label, un_z, num_examples):
        indices = np.arange(len(un_label))        
        target_indices = (un_z == self.target_group[1]) 
        logits = model.predict_proba(un_data[target_indices])
        
        r_arr = - np.abs(logits[:,self.target_group[0]] - (1 - self.r))
        
        selection = np.argsort(r_arr)[::-1][:num_examples]
        temp_indices = indices[target_indices][selection]
        sel_indices = temp_indices[un_label[temp_indices] == self.target_group[0]]
        
        return sel_indices, temp_indices
        
class Falcon():
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, target_fairness, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        
        self.target_fairness = target_fairness
        self.ml_method = ml_method
    
        self.mabs = dict()

    def make_query(self, prev_model, num_examples, target_groups, policy_set):
        key = str(target_groups)
        if key in self.mabs:
            target_mab = self.mabs[key]
            target_mab.update(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z)
        else:
            target_mab = MAB(self.train_data, self.train_label, self.train_z, self.un_data, self.un_label, self.un_z, self.val_data, self.val_label, self.val_z, target_groups, policy_set, self.target_fairness, self.ml_method)
            self.mabs[key] = target_mab
                
        model = target_mab.make_query(prev_model, num_examples)
        self.update(target_mab.train_data, target_mab.train_label, target_mab.train_z, target_mab.un_data, target_mab.un_label, target_mab.un_z)
        return model
        
    def update(self, train_data, train_label, train_z, un_data, un_label, un_z):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z

class MAB():
    def __init__(self, train_data, train_label, train_z, un_data, un_label, un_z, val_data, val_label, val_z, target_groups, policy_set, target_fairness, ml_method):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z
        self.val_data = val_data
        self.val_label = val_label
        self.val_z = val_z
        
        self.target_groups = target_groups
        self.target_fairness = target_fairness
        self.ml_method = ml_method
        
        self.query_polices = []
        for target_group in self.target_groups:
            for policy in policy_set:
                r = float(policy[2:])
                self.query_polices.append([Single_policy(target_group = target_group, r=r), target_group])

        self.K = len(self.query_polices)
        self.w = np.array([1. for _ in range(self.K)])       
        self.iter, self.warm_step, self.gamma = 0, 100, 1
        self.max_val, self.max = 0, 1
            
    def cal_reward(self, model, prev_model, num_examples):
        value, _ = fairness_metrics(model, self.val_data, self.val_label, self.val_z, self.target_fairness)
        prev_value, _ = fairness_metrics(prev_model, self.val_data, self.val_label, self.val_z, self.target_fairness)
        raw_value = (value - prev_value)
        
        if raw_value > self.max_val:
            self.max_val = raw_value

        if (self.iter * num_examples) == self.warm_step:
            if self.max_val != 0:
                self.max = self.max_val
                g = 1 / (self.max)
                self.gamma = np.min([1, np.sqrt(self.K * np.log(self.K) / (np.exp(1)-1) / g)])
        elif (self.iter * num_examples) > self.warm_step:
            if self.max == 1 and self.max_val != 0:
                self.max = self.max_val
                g = 1 / (self.max)
                self.gamma = np.min([1, np.sqrt(self.K * np.log(self.K) / (np.exp(1)-1) / g)])
                
        if raw_value >= 0:
            reward = raw_value / self.max
        else:
            reward = 0
            
        return reward

    def make_query(self, prev_model, num_examples):
        self.iter += 1
        
        self.p = (1 - self.gamma) * self.w / np.sum(self.w) + self.gamma / self.K
        policy_id = np.random.choice(np.arange(self.K), size=1, p=self.p, replace=False)[0]
        policy, target_group = self.query_polices[policy_id]
        sel_indices, temp_indices = policy.make_query(prev_model, self.un_data, self.un_label, self.un_z, num_examples)
        
        self.train_data = np.concatenate((self.train_data, self.un_data[sel_indices]), axis=0)
        self.train_label = np.concatenate((self.train_label, self.un_label[sel_indices]), axis=0)
        self.train_z = np.concatenate((self.train_z, self.un_z[sel_indices]), axis=0)
        
        model = model_training(self.train_data, self.train_label, self.val_data, self.val_label, self.ml_method)
        reward = self.cal_reward(model, prev_model, num_examples)
            
        self.un_data = np.delete(self.un_data, temp_indices, axis=0)
        self.un_label = np.delete(self.un_label, temp_indices, axis=0)
        self.un_z = np.delete(self.un_z, temp_indices, axis=0)
        
        if (self.iter * num_examples >= self.warm_step):
            rhat = np.zeros(self.K)
            exp_reward = reward / self.p[policy_id]
            for i in range(self.K):
                if (self.query_polices[policy_id][1] == self.query_polices[i][1]) and (policy_id - 1 <= i <= policy_id + 1):
                    rhat[i] = exp_reward * (1 - np.abs(i - policy_id) * 0.5)
                    
            self.w = self.w * np.exp(self.gamma * rhat / self.K)
            
        return model
                
    def update(self, train_data, train_label, train_z, un_data, un_label, un_z):
        self.train_data = train_data
        self.train_label = train_label
        self.train_z = train_z
        self.un_data = un_data
        self.un_label = un_label
        self.un_z = un_z