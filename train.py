import sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import DRL_Model, DQN
from utils import DataInput, evaluate, update_target_graph

# Note: this code must be run using tensorflow 1.4.0
tf.reset_default_graph()

# Hyperparameter Settings
random.seed(625)
np.random.seed(625)
tf.set_random_seed(625)
batch_size = 32
hidden_size = 128
learning_rate = 1
session_length = 10
epoch = 1000

# Loading Dataset
data = pd.read_csv('data.txt', low_memory=False)
data.columns = ['user_id','video_id','click','hour']
userid = list(set(data['user_id']))
videoid = list(set(data['video_id']))
user_count = len(userid)
video_count = len(videoid)

# Splitting Dataset into Training Set and Test Set for Cross-Validation
data = data.sample(frac=1)
validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]

# Obtaining Session-Based User Interaction Records
train_set, test_set = [], []
for user in userid:
    train_user = train_data.loc[train_data['user_id']==user]
    train_user = train_user.sort_values(['hour'])
    length = len(train_user)
    train_user.index = range(length)
    if length > session_length+1:
        for i in range(length-session_length-1):
            train_set.append((user, train_user.loc[i+session_length-1,'video_id'], train_user.loc[i+session_length,'video_id'], list(train_user.loc[i:i+session_length-1,'video_id']), list(train_user.loc[i+1:i+session_length,'video_id']), float(train_user.loc[i+session_length,'click'])))
    test_user = test_data.loc[test_data['user_id']==user]
    test_user = test_user.sort_values(['hour'])
    length = len(test_user)
    test_user.index = range(length)
    if length > session_length+1:
        for i in range(length-session_length-1):
            test_set.append((user, test_user.loc[i+session_length-1,'video_id'], test_user.loc[i+session_length,'video_id'], list(test_user.loc[i:i+session_length-1,'video_id']), list(test_user.loc[i+1:i+session_length,'video_id']), float(test_user.loc[i+session_length,'click'])))
random.shuffle(train_set)
random.shuffle(test_set)
train_set = train_set[:len(train_set)//batch_size*batch_size]
test_set = test_set[:len(test_set)//batch_size*batch_size]

# Training the DRLTRS Model
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    primary_session_dqn = DQN(hidden_size, 'primary_session_dqn')
    target_session_dqn = DQN(hidden_size, 'target_session_dqn')
    primary_record_dqn = DQN(hidden_size, 'primary_record_dqn')
    target_record_dqn = DQN(hidden_size, 'target_record_dqn')
    model = DRL_Model(user_count, video_count, hidden_size, batch_size, primary_session_dqn, target_session_dqn, primary_record_dqn, target_record_dqn)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    auc, hit_rate, _ = evaluate(sess, model, test_set)
    print('Initial AUC: %.4f\t Hit Rate: %.4f' % (auc, hit_rate))
    sys.stdout.flush()
    start_time = time.time()
    
    for _ in range(epoch):
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, learning_rate)
            loss_sum += loss
        print('Epoch %d Train_Loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))      
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        auc, hit_rate, _ = evaluate(sess, model, test_set)
        print('AUC: %.4f\t Hit Rate: %.4f' % (auc, hit_rate))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        update_target_graph('primary_session_dqn','target_session_dqn')
        update_target_graph('primary_record_dqn','target_record_dqn')