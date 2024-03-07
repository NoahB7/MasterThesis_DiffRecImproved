import numpy as np
import random
import pandas as pd
import scipy.sparse as sp


# data = pd.read_csv('datasets/archive/Books_rating.csv')

# data = data.drop(columns=['Price','profileName','review/helpfulness','review/summary','review/text'])

# data = data[data['review/score'] >= 4.0]

# for col in data.columns:
#     bools = pd.isnull(data[col])
#     if bools.any():
#         print('nulls found in ', col)
#         print(len(data[bools]))

# titlebools = pd.notnull(data['Title'])
# idbools = pd.notnull(data['User_id'])

# bools = [a and b for a, b in zip(titlebools, idbools)]
# data = data[bools]

# print(data)


# ----------------------------------------------------------------------------------------------------------------------------------------------

# data = pd.read_csv('datasets/ml-1m_custom/ratings.dat', names=['stuff'])

# data[['UID','MID','R','T']] = data['stuff'].str.split('::', expand=True) # convert :: delimited data
# data = data.drop(columns = ['stuff']) # remove non numeric data
# for col in data.columns:
#     data[col] = pd.to_numeric(data[col]) # convert all numeric data to integers

# data = data[data['R'] >= 4] # remove ratings 3 and below

# data = data.sort_values(['UID','T'])
# data = data.drop(columns = ['T', 'R'])
# data = np.array(data)

# userids = {}
# ruserids = {}
# movieids = {}
# rmovieids = {}
# ucount = 0
# mcount = 0
# count = 0
# for uid, mid in data:
#     if uid not in userids:
#         userids[uid] = ucount
#         ucount+=1
#         data[count,0] = userids[uid]
#     else:
#         data[count,0] = userids[uid]

#     if mid not in movieids:
#         movieids[mid] = mcount
#         mcount+=1
#         data[count,1] = movieids[mid]
#     else:
#         data[count,1] = movieids[mid]
#     count += 1


# np.save('./datasets/ml-1m_custom/train_list.npy', data, allow_pickle=True)
# np.save('./datasets/ml-1m_custom/test_list.npy', data, allow_pickle=True)
# np.save('./datasets/ml-1m_custom/valid_list.npy', data, allow_pickle=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------

train_list = np.load('./datasets/ml-1m_custom/train_list.npy', allow_pickle=True)
train_list1 = np.load('./datasets/ml-1m_clean/train_list.npy', allow_pickle=True)

print(train_list[0:50])
print(train_list1[0:50])

# print(train_list[40:200])
# uid_max = 0
# iid_max = 0
# train_dict = {}
# for uid, iid in train_list:
#         if uid not in train_dict:
#             train_dict[uid] = []
#         train_dict[uid].append(iid)
#         if uid > uid_max:
#             uid_max = uid
#         if iid > iid_max:
#             iid_max = iid
    
# # add one to get count since ids start at 0
# n_user = uid_max + 1
# n_item = iid_max + 1
# print(f'user num: {n_user}')
# print(f'item num: {n_item}')

# # original data is in shape  (429993 , 2) for 429,993 reviews of a user a on movie b

# # np.ones_like is just all 1's (a place holder I think, not sure why it has to be done this way)

# # the operations below transform the 429k reviews into the sparse matrix format of user 
# train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
#                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
#                 shape=(n_user, n_item))
#                 #  shape=(n_user, n_item))

# print(train_data[0:50])