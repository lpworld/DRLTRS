import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')
    
import sys, time, random
import tensorflow as tf
from model import DRL_Model, DQN
from utils import DataInput, evaluate, update_target_graph, recommendation, compute_gini, compute_diversity

# Note: this code must be run using tensorflow 1.4.0
tf.reset_default_graph()

# Hyperparameter Settings
num_user = 400
num_item = 400
time_range = 24
batch_size = 32
hidden_size = 128
epoch = 50
learning_rate = 1
session_length = 10

# Generate Synthetic Data
ratings, users, items = [], [], []
rating_matrix = np.zeros((num_user,num_item))
for u in range(num_user):
    ru  = np.random.normal(0.5,1,1)
    for i in range(num_item):
        ri = np.random.normal(0.5,0.5,1)
        eij = np.random.normal(0,0.1,1)
        r = ru+ri+eij
        rating = max(min(r[0],1),0)
        rating = int(rating>0.5)
        rating_matrix[u][i] = rating
        ratings.append(rating)
        users.append(u)
        items.append(i)
times = np.random.randint(1,time_range,size=num_user*num_item)
data = pd.DataFrame({'user_id':users,'item_id':items,'click':ratings,'time':times})
data = data.sort_values(['time']) #simulate time records

# plitting Dataset into Training Set and Test Set for Cross-Validation
validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]

# Obtaining Session-Based User Interaction Records
train_set, test_set, gini_item_set = [], [], []
for user in range(num_user):
    train_user = train_data.loc[train_data['user_id']==user]
    length = len(train_user)
    train_user.index = range(length)
    if length > session_length+1:
        for i in range(length-session_length-1):
            train_set.append((user, train_user.loc[i+session_length-1,'item_id'], train_user.loc[i+session_length,'item_id'], list(train_user.loc[i:i+session_length-1,'item_id']), list(train_user.loc[i+1:i+session_length,'item_id']), float(train_user.loc[i+session_length,'click'])))
for user in range(num_user):
    test_user = test_data.loc[test_data['user_id']==user]
    length = len(test_user)
    sub_item = list(test_user['item_id'])
    sub_item = random.choices(sub_item,k=session_length+1)
    gini_item_set = gini_item_set + sub_item
    test_set.append((user, sub_item[session_length-1], sub_item[session_length], sub_item[0:session_length], sub_item[1:session_length+1], 1.0))
random.shuffle(train_set)
random.shuffle(test_set)
train_set = train_set[:len(train_set)//batch_size*batch_size]
test_set = test_set[:len(test_set)//batch_size*batch_size]

# Training the DRLTRS Model
i_table = list(range(num_item))
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    primary_session_dqn = DQN(hidden_size, 'primary_session_dqn')
    target_session_dqn = DQN(hidden_size, 'target_session_dqn')
    primary_record_dqn = DQN(hidden_size, 'primary_record_dqn')
    target_record_dqn = DQN(hidden_size, 'target_record_dqn')
    model = DRL_Model(num_user, num_item, hidden_size, batch_size, primary_session_dqn, target_session_dqn, primary_record_dqn, target_record_dqn)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    sys.stdout.flush()
    start_time = time.time()
    
    for _ in range(epoch):
        item_embedding = []
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / batch_size)
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, learning_rate)    
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        item_emb_table = model.get_item_emb(sess, i_table)

        #Compute AUC Metrics Here
        auc, hit_rate, _ = evaluate(sess, model, train_set)
        _, _, embedding_table = evaluate(sess, model, test_set)
        print('AUC: %.4f\t' % auc)
        
        #Update Training Set and Compute Gini Coefficients Here
        newtrain_set, new_gini_set = recommendation(embedding_table, item_emb_table)
        train_set = train_set + newtrain_set
        gini_item_set = gini_item_set + new_gini_set
        gini = compute_gini(gini_item_set)
        print('GINI: %.4f\t' % gini)
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        update_target_graph('primary_session_dqn','target_session_dqn')
        update_target_graph('primary_record_dqn','target_record_dqn')
