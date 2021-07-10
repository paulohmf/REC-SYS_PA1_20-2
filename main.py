import sys
import pandas as pd
import numpy as np
#import time
#time_begin = time.time()  # Time

# Getting entry files
ratings_file = sys.argv[1]
targets_file = sys.argv[2]

# Load train
ratings_df = pd.read_csv(ratings_file)

mean = ratings_df['Prediction'].mean()

# Train manipulation
ratings_df['UserId'] = ratings_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[0][1:]))
ratings_df['ItemId'] = ratings_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[1][1:]))
ratings_df = ratings_df[['UserId','ItemId','Prediction']]

# Train analysis 
ratings_df['Counts Users'] = ratings_df.groupby(['UserId'])['ItemId'].transform('count')
ratings_df['Counts Items'] = ratings_df.groupby(['ItemId'])['UserId'].transform('count')
ratings_df["id"] = ratings_df.index

## Train cleaning
user_based_df = ratings_df[ratings_df['Counts Users'] >= 5] 
item_based_df = ratings_df[ratings_df['Counts Items'] >= 5]
del ratings_df

## User based recommendation
# Generating lookup tables 
unique_users = pd.unique(user_based_df['UserId'])
unique_items = pd.unique(user_based_df['ItemId'])

i = 0
U_lookup_users = dict()
for u in unique_users:
    U_lookup_users[u] = i
    i += 1

j = 0
U_lookup_itens = dict()
for t in unique_items:
    U_lookup_itens[t] = j
    j += 1

# Generating ratings matrix
ratings_matrix_user = np.zeros((len(unique_users),len(unique_items)))
for u,i,r in zip(user_based_df['UserId'].values, user_based_df['ItemId'].values, user_based_df['Prediction'].values):
  ratings_matrix_user[U_lookup_users[u]][U_lookup_itens[i]] = r

## Calculating user similarity matrix
norm = np.reshape(np.linalg.norm(ratings_matrix_user, axis=1), (1, ratings_matrix_user.shape[0]))
n2 = norm * norm.T
c = ratings_matrix_user @ ratings_matrix_user.T
user_similarity_matrix = c / n2

## Item based recommendation
# Generating lookup tables 
unique_users = pd.unique(item_based_df['UserId'])
unique_items = pd.unique(item_based_df['ItemId'])

i = 0
I_lookup_users = dict()
for u in unique_users:
    I_lookup_users[u] = i
    i += 1

j = 0
I_lookup_itens = dict()
for t in unique_items:
    I_lookup_itens[t] = j
    j += 1

# Generating ratings matrix
ratings_matrix_item = np.zeros((len(unique_users),len(unique_items)))
for u,i,r in zip(item_based_df['UserId'].values, item_based_df['ItemId'].values, item_based_df['Prediction'].values):
  ratings_matrix_item[I_lookup_users[u]][I_lookup_itens[i]] = r

## Calculating item similarity matrix
ratings_matrix_item = ratings_matrix_item.T
norm = np.reshape(np.linalg.norm(ratings_matrix_item, axis=1), (1, ratings_matrix_item.shape[0]))
n2 = norm * norm.T
c = ratings_matrix_item @ ratings_matrix_item.T
item_similarity_matrix = c / n2
ratings_matrix_item = ratings_matrix_item.T

# Target Load and Manipulation
targets_df = pd.read_csv(targets_file)
targets_df['UserId'] = targets_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[0][1:]))
targets_df['ItemId'] = targets_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[1][1:]))
targets_df = targets_df[['UserId','ItemId']]

# Recommendation
np.seterr(divide='ignore', invalid='ignore')
f = open("results.csv", "w")
f.write("UserId:ItemId,Prediction\n")

for u,i in zip(targets_df['UserId'].values,targets_df['ItemId'].values):
  # User based rate
  u_rate = -1.0
  if u in U_lookup_users and i in U_lookup_itens:
    m_u = U_lookup_users[u]
    m_i = U_lookup_itens[i]

    ratings_array = ratings_matrix_user[:,m_i]
    similarity_array = user_similarity_matrix[m_u,:]

    pos_array = np.where(ratings_array != 0.0)
    norm_similarity = [similarity_array[pos] for pos in pos_array]

    u_rate = similarity_array@ratings_array/np.sum(norm_similarity)
    if np.isnan(u_rate):
      u_rate = -1.0 

  # Item based rate
  i_rate = -1.0
  if u in I_lookup_users and i in I_lookup_itens:
    m_u = I_lookup_users[u]
    m_i = I_lookup_itens[i]

    ratings_array = ratings_matrix_item[m_u,:]
    similarity_array = item_similarity_matrix[:,m_i]

    pos_array = np.where(ratings_array != 0.0)
    norm_similarity = [similarity_array[pos] for pos in pos_array]

    i_rate = similarity_array@ratings_array/np.sum(norm_similarity)
    if np.isnan(i_rate):
      i_rate = -1.0 

  # Mixing rates
  if u_rate != -1.0 and i_rate != -1.0:
    rate = (u_rate + i_rate)/2
  elif u_rate != -1.0:
    rate = u_rate
  elif i_rate != -1.0:
    rate = i_rate
  else:
    rate = mean

  string ="u{:07d}".format(u) + ":" + "i{:07d}".format(i) + "," + str(rate) + "\n"
  f.write(string)

f.close()

#print(time.time() - time_begin)  # Time