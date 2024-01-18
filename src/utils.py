import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, generate_categories, ACSTravelTime, ACSEmployment, ACSIncome
from aif360.datasets import CompasDataset

def fairness_metrics(clf, data, label, z, fairness_type):
    """
        Obatains fairness score and corresponding target subgroups.
    """
    y_hat = clf.predict(data)
    
    num_groups = len(np.unique(z))
    if fairness_type == "DP":
        DPs = []
        for z_value in range(num_groups):
            z_mask = (z == z_value)    
            Pr_y_hat_1_z = np.sum((y_hat == 1)[z_mask]) / np.sum(z_mask)
            DPs.append(Pr_y_hat_1_z)
        
        max_group, min_group = np.argmax(DPs), np.argmin(DPs)
        target_groups = [(1, min_group), (0, max_group)]
        metric = DPs[max_group] - DPs[min_group]

    elif fairness_type == "EO":
        FNRs = []
        for z_value in range(num_groups):
            y_1_z_mask = (label == 1) & (z == z_value)
            Pr_y_hat_1_y_1_z = np.sum((y_hat == 1)[y_1_z_mask]) / np.sum(y_1_z_mask)
            FNRs.append(Pr_y_hat_1_y_1_z)
            
        max_group, min_group = np.argmax(FNRs), np.argmin(FNRs)
        target_groups = [(1, min_group)]
        metric = FNRs[max_group] - FNRs[min_group]
        
    elif fairness_type == "ED":
        FPRs, FNRs = [], []
        for z_value in range(num_groups):
            y_1_z_mask = (label == 1) & (z == z_value)
            Pr_y_hat_1_y_1_z = np.sum((y_hat == 1)[y_1_z_mask]) / np.sum(y_1_z_mask)
            FNRs.append(Pr_y_hat_1_y_1_z)
            
            y_0_z_mask = (label == 0) & (z == z_value)
            Pr_y_hat_1_y_0_z = np.sum((y_hat == 1)[y_0_z_mask]) / np.sum(y_0_z_mask)
            FPRs.append(Pr_y_hat_1_y_0_z)
            
        FPR_diff = max(FPRs) - min(FPRs)
        FNR_diff = max(FNRs) - min(FNRs)
        if FPR_diff >= FNR_diff:
            max_group, min_group = np.argmax(FPRs), np.argmin(FPRs)
            target_groups = [(0, max_group)]
            metric = FPR_diff
        else:
            max_group, min_group = np.argmax(FNRs), np.argmin(FNRs)
            target_groups = [(1, min_group)]
            metric = FNR_diff
            
    elif fairness_type == "PP":
        FORs, FDRs = [], []
        for z_value in range(num_groups):
            yhat_0_z_mask = (y_hat == 0) & (z == z_value)
            Pr_y_1_yhat_0_z = np.sum((label == 1)[yhat_0_z_mask]) / np.sum(yhat_0_z_mask)
            FORs.append(Pr_y_1_yhat_0_z)
            
            yhat_1_z_mask = (y_hat == 1) & (z == z_value)
            Pr_y_1_yhat_1_z = np.sum((label == 1)[yhat_1_z_mask]) / np.sum(yhat_1_z_mask)
            FDRs.append(Pr_y_1_yhat_1_z)
            
        FOR_diff = max(FORs) - min(FORs)
        FDR_diff = max(FDRs) - min(FDRs)
        if FOR_diff >= FDR_diff:
            max_group, min_group = np.argmax(FORs), np.argmin(FORs)
            target_groups = [(0, max_group), (1, max_group)]
            metric = FOR_diff
        else:
            max_group, min_group = np.argmax(FDRs), np.argmin(FDRs)
            target_groups = [(0, min_group), (1, min_group)]
            metric = FDR_diff

    elif fairness_type == "EER":
        EERs = []
        for z_value in range(num_groups):
            z_mask = (z == z_value)
            Pr_yhat_y_z = np.sum((label != y_hat)[z_0_mask]) / np.sum(z_mask)
            EERs.append(Pr_yhat_y_z)
            
        max_group, min_group = np.argmax(FDRs), np.argmin(FDRs)
        target_groups = [(0, max_group), (1, max_group)]
        metric = EERs[max_group] - EERs[min_group]

    fairness_score = 1 - metric
    return fairness_score, target_groups

def generate_dataset(data, label, z, train_data_num, test_data_num, un_data_num, val_data_num, seed):    
    """
        Constructs train, test, unlabeled, and validation datasets.
    """
    data, label, z = random_shuffle(data, label, z, seed)
    overall_indices = np.arange(len(label))
    
    num_groups = len(np.unique(z))
    for i in range(num_groups):
        for j in range(2):
            feature_index = 2*i+j

            train_indices = overall_indices[(z == i) & (label == j)][:train_data_num[feature_index]]
            test_indices = overall_indices[(z == i) & (label == j)][train_data_num[feature_index]:train_data_num[feature_index]+test_data_num[feature_index]]
            un_indices = overall_indices[(z == i) & (label == j)][train_data_num[feature_index]+test_data_num[feature_index]:train_data_num[feature_index]+test_data_num[feature_index]+un_data_num[feature_index]]
            val_indices = overall_indices[(z == i) & (label == j)][train_data_num[feature_index]+test_data_num[feature_index]+un_data_num[feature_index]:train_data_num[feature_index]+test_data_num[feature_index]+un_data_num[feature_index]+val_data_num[feature_index]]

            if feature_index == 0:
                x_train = data[train_indices]
                y_train = label[train_indices]
                z_train = z[train_indices]

                x_test = data[test_indices]
                y_test = label[test_indices]
                z_test = z[test_indices]

                x_un = data[un_indices]
                y_un = label[un_indices]
                z_un = z[un_indices]

                x_val = data[val_indices]
                y_val = label[val_indices]
                z_val = z[val_indices]
            else:
                x_train = np.concatenate((x_train, data[train_indices]), axis=0)
                y_train = np.concatenate((y_train, label[train_indices]), axis=0)
                z_train = np.concatenate((z_train, z[train_indices]), axis=0)

                x_test = np.concatenate((x_test, data[test_indices]), axis=0)
                y_test = np.concatenate((y_test, label[test_indices]), axis=0)
                z_test = np.concatenate((z_test, z[test_indices]), axis=0)

                x_un = np.concatenate((x_un, data[un_indices]), axis=0)
                y_un = np.concatenate((y_un, label[un_indices]), axis=0)
                z_un = np.concatenate((z_un, z[un_indices]), axis=0)

                x_val = np.concatenate((x_val, data[val_indices]), axis=0)
                y_val = np.concatenate((y_val, label[val_indices]), axis=0)
                z_val = np.concatenate((z_val, z[val_indices]), axis=0)

    x_train, y_train, z_train = random_shuffle(x_train, y_train, z_train, seed)
    x_test, y_test, z_test = random_shuffle(x_test, y_test, z_test, seed)
    x_un, y_un, z_un = random_shuffle(x_un, y_un, z_un, seed)
    x_val, y_val, z_val = random_shuffle(x_val, y_val, z_val, seed)
    
    return x_train, y_train, z_train, x_test, y_test, z_test, x_un, y_un, z_un, x_val, y_val, z_val

def load_dataset(dataset_type):
    """
        Load dataset from Folktables and AIF360.
    """
    scaler = StandardScaler()

    if dataset_type == "TravelTime":
        dataset = ACSTravelTime

        data_source = ACSDataSource(root_dir="../data", survey_year='2018', horizon='1-Year', survey='person')
        definition_df = data_source.get_definitions(download=True)
        categories = generate_categories(features=dataset.features, definition_df=definition_df)

        acs_data = data_source.get_data(states=["CA"], download=True)
        ca_features, ca_labels, _ = dataset.df_to_pandas(acs_data, categories=categories, dummies=True)

        female_index = ca_features.columns.get_loc("SEX_Female")
        male_index = ca_features.columns.get_loc("SEX_Male")

        data = ca_features.to_numpy(dtype="float")
        data[:,:4] = scaler.fit_transform(data[:,:4])
        label = ca_labels.to_numpy().ravel()
        z = data[:, male_index]

        train_data_num = [1115, 181, 441, 709]
        test_data_num = [11150, 1815, 4410, 7095]
        un_data_num = [22300, 3630, 8820, 14190]
        val_data_num = [1115, 181, 441, 709]
        group_names = ["Female-0", "Female-1", "Male-0", "Male-1"]

    elif dataset_type == "Employ":
        dataset = ACSEmployment

        data_source = ACSDataSource(root_dir="../data", survey_year='2018', horizon='1-Year', survey='person')
        definition_df = data_source.get_definitions(download=True)
        categories = generate_categories(features=dataset.features, definition_df=definition_df)

        acs_data = data_source.get_data(states=["CA"], download=True)
        ca_features, ca_labels, _ = dataset.df_to_pandas(acs_data, categories=categories, dummies=True)

        dis_index = ca_features.columns.get_loc("DIS_With a disability")
        able_index = ca_features.columns.get_loc("DIS_Without a disability")

        data = ca_features.to_numpy(dtype="float")
        data[:,:1] = scaler.fit_transform(data[:,:1])
        label = ca_labels.to_numpy().ravel()
        z = data[:, able_index]

        train_data_num = [579, 81, 1673, 3292]
        test_data_num = [8685, 1215, 25095, 49380]
        un_data_num = [17370, 2430, 50190, 98760]
        val_data_num = [579, 81, 1673, 3292]
        group_names = ["Disability-0", "Disability-1", "Able_bodied-0", "Able_bodied-1"]

    elif dataset_type == "Income":
        race_list = ["White alone", "Black or African American alone", "American Indian alone", "Alaska Native alone", 
                 "American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races",
                 "Asian alone", "Native Hawaiian and Other Pacific Islander alone", "Some Other Race alone", "Two or More Races"]

        dataset = ACSIncome

        data_source = ACSDataSource(root_dir="../data", survey_year='2018', horizon='1-Year', survey='person')
        definition_df = data_source.get_definitions(download=True)
        categories = generate_categories(features=dataset.features, definition_df=definition_df)

        acs_data = data_source.get_data(states=["CA"], download=True)
        ca_features, ca_labels, _ = dataset.df_to_pandas(acs_data, categories=categories, dummies=True)

        white_index = ca_features.columns.get_loc("RAC1P_White alone")
        asian_index = ca_features.columns.get_loc("RAC1P_Asian alone")

        data = ca_features.to_numpy(dtype="float")
        white_col = np.array(data[:, white_index])
        asian_col = np.array(data[:, asian_index])
        others_col = 1 - (np.array(data[:, white_index]) + np.array(data[:, asian_index]))

        remove_indices = []
        for race in race_list:
            remove_indices.append(ca_features.columns.get_loc("RAC1P_"+ race))
        data = np.delete(data, remove_indices, axis=1)

        data[:,:2] = scaler.fit_transform(data[:,:2])
        data = np.append(data, np.reshape(others_col, (-1, 1)), axis=1)
        data = np.append(data, np.reshape(white_col, (-1, 1)), axis=1)
        data = np.append(data, np.reshape(asian_col, (-1, 1)), axis=1)
        label = ca_labels.to_numpy().ravel()
        z = white_col + 2 * asian_col

        train_data_num = [309, 109, 2019, 268, 169, 314]
        test_data_num = [3090, 1090, 20190, 2680, 1690, 3140]
        un_data_num = [6180, 2180, 40380, 5360, 3380, 6280]
        val_data_num = [309, 109, 2019, 268, 169, 314]
        group_names = ["Others-0", "Others-1", "White-0", "White-1", "Asian-0", "Asian-1"]

    elif dataset_type == "COMPAS":
        dataset = CompasDataset(label_name='two_year_recid', favorable_classes=[0], 
                            protected_attribute_names=['sex'], privileged_classes=[['Female']], 
                            categorical_features=['age_cat', 'c_charge_degree', 'c_charge_desc', 'race'], 
                            features_to_keep=['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'two_year_recid'], 
                            features_to_drop=[], na_values=[])

        male_index = dataset.feature_names.index("sex")

        data = dataset.features
        data[:,1:2] = scaler.fit_transform(data[:,1:2])
        label = dataset.labels.ravel()
        z = data[:,male_index]

        train_data_num = [86, 158, 37, 13]
        test_data_num = [344, 632, 150, 52]
        un_data_num = [688, 1264, 300, 104]
        val_data_num = [86, 158, 37, 13]
        group_names = ["Female-0", "Female-1", "Male-0", "Male-1"]
        
    return data, label, z, group_names, train_data_num, test_data_num, un_data_num, val_data_num
    
def model_training(x_train, y_train, x_val, y_val, ml_method):
    """
        Trains ML models.
    """
    if ml_method == "lr":
        model = LogisticRegression(solver='lbfgs', C=1).fit(x_train, y_train)
    elif ml_method == "nn":        
        model = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(10), random_state=seed).fit(x_train, y_train)
    return model

def random_shuffle(data, label, z, seed):
    """
        Randomly shuffle the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    shuffle = np.arange(len(data))
    np.random.shuffle(shuffle)
    data = data[shuffle]
    label = label[shuffle]
    z = z[shuffle]
    
    return data, label, z

def cal_subnum(data, label, z):
    """
        Computes the number of samples for each subgroup.
    """
    sub_num = []
    
    num_groups = len(np.unique(z))
    for i in range(num_groups):
        for j in range(2):
            indices = (z == i) & (label == j)
            sub_num.append(np.sum(indices))

    return sub_num
