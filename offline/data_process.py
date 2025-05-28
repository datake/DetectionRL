import pandas as pd
import numpy as np
import os

path = os.getcwd()
path += '/original_data_without_random_seed/'
# path += '/ppo_M_W_without_random_seed/'
# path += '/ppo_M_without_random_seed/'
dir_names = os.listdir(path)

ppo_paper_results = pd.read_csv('ppo_paper_results.csv')

higher_count =0
lower_count = 0
same_count = 0
for dir in dir_names:
    if 'mean_score_std' in dir:
        game_name = dir.split('NoFrameskip')[0]
        game_index = np.where(ppo_paper_results['Game']==game_name)[0][0]
        paper_result = float(ppo_paper_results['PPO'][game_index])

        data = pd.read_csv(path + dir)
        mean_score = np.mean(data['mean_score'][-3:])

        if mean_score > (paper_result + abs(paper_result*.2)):
            print('higher :',game_name,paper_result,round(mean_score,1))
            higher_count +=1
        elif mean_score < (paper_result - abs(paper_result*.2)):
            lower_count +=1
            print('lower :',game_name,paper_result,round(mean_score,1))
        else:
            same_count += 1
            print('same :',game_name,paper_result,round(mean_score,1))
print(lower_count,same_count,higher_count)





