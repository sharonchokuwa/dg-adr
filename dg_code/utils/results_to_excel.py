import pandas as pd
import os

def initialize_excel():
    #store the data
    data = pd.DataFrame(columns=['timestamp', 'description', 
                                'acc1', 'auc1', 'f1-score1',
                                'acc2', 'auc2', 'f1-score2',
                                'acc3', 'auc3', 'f1-score3',
                                'acc4', 'auc4', 'f1-score4',
                                'acc5', 'auc5', 'f1-score5',
                                'acc6', 'auc6', 'f1-score6',
                                'acc7', 'auc7', 'f1-score7',
                                'avg_acc', 'avg_auc', 'avg_f1-score'])
    return data
    

#calculate average metrics
def calculate_average(data):
    avg_acc = data[['acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7']].mean().mean()
    avg_auc = data[['auc1', 'auc2', 'auc3', 'auc4', 'auc5', 'auc6', 'auc7']].mean().mean()
    avg_f1_score = data[['f1-score1', 'f1-score2', 'f1-score3', 'f1-score4', 'f1-score5', 'f1-score6', 'f1-score7']].mean().mean()
    return avg_acc, avg_auc, avg_f1_score

def add_data(data, model_number, metrics):
    acc = metrics['acc']
    auc = metrics['auc']
    f1_score = metrics['f1'] 

    # Create column names based on the model number
    acc_col = f'acc{model_number}'
    auc_col = f'auc{model_number}'
    f1_col = f'f1-score{model_number}'
    data.loc[0, [acc_col, auc_col, f1_col]] = [acc, auc, f1_score]
    return data

def save_to_excel(log_path, data, cfg):
    timestamp = cfg.TIMESTAMP
    description = cfg.DESCRIPTION
    avg_acc, avg_auc, avg_f1_score = calculate_average(data)
    data.loc[0, ['timestamp', 'description', 'avg_acc', 'avg_auc', 'avg_f1-score']] = [timestamp, description, avg_acc, avg_auc, avg_f1_score]
    path = os.path.join('result', timestamp, cfg.ALGORITHM, f'{timestamp}.xlsx')
    data.to_excel(path, index=False)
