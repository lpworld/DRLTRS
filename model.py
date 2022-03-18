import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from utils import DataInput
import numpy as np

class DRL_Model(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, primary_session_dqn, target_session_dqn, primary_record_dqn, target_record_dqn):

        # Hyperparameter Settings
        hidden_size = 128
        session_length = 10
        gamma = 0.9
        margin = 0.1
        
        # Tensorflow Parameter Placeholders
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.last_i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, session_length]) # [B, T]
        self.next_hist = tf.placeholder(tf.int32, [batch_size, session_length]) # [B, T]
        self.label = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])
        self.i_table = tf.placeholder(tf.int32, [item_count,]) # [I]
        self.primary_session_dqn = primary_session_dqn
        self.target_session_dqn = target_session_dqn
        self.primary_record_dqn = primary_record_dqn
        self.target_record_dqn = target_record_dqn

        # Embedding Table Initialization
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size])
        user_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        item_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        last_item_emb = tf.nn.embedding_lookup(item_emb_w, self.last_i)
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist)
        next_h_emb = tf.nn.embedding_lookup(item_emb_w, self.next_hist)
        item_emb_table = tf.nn.embedding_lookup(item_emb_w, self.i_table)

        # Session-Level DQNs for Producing Session Policy
        session_policy = tf.layers.dense(user_emb, 80, activation=tf.nn.sigmoid)
        session_policy = tf.layers.dense(session_policy, 40, activation=tf.nn.sigmoid)
        session_policy = tf.layers.dense(session_policy, hidden_size, activation=None)

        # Self-Attnetive GRU Network for Constructing Consumer State Representations
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
        	output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        	state, alphas = self.seq_attention(output, hidden_size, session_length)
        	state = tf.nn.dropout(state, 0.1)

        	next_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float32)
        	next_state, next_alphas = self.seq_attention(next_output, hidden_size, session_length)
        	next_state = tf.nn.dropout(next_state, 0.1)

        # Record-Level DQNs for Producing Actions of Transition Vectors
        with tf.variable_scope('transition', reuse=tf.AUTO_REUSE):
            transition = tf.layers.batch_normalization(inputs=tf.concat([session_policy, state], axis=1))
            transition = tf.layers.dense(transition, 80, activation=tf.nn.sigmoid)
            transition = tf.layers.dense(transition, 40, activation=tf.nn.sigmoid)
            transition = tf.layers.dense(transition, hidden_size, activation=None)
            
            next_transition = tf.layers.batch_normalization(inputs=tf.concat([session_policy, next_state], axis=1))
            next_transition = tf.layers.dense(next_transition, 80, activation=tf.nn.sigmoid)
            next_transition = tf.layers.dense(next_transition, 40, activation=tf.nn.sigmoid)
            next_transition = tf.layers.dense(next_transition, hidden_size, activation=None)

        # Computing the Metric Loss for the Predicted Target Embeddings
        target_item_emb = tf.add(last_item_emb, transition)
        self.distance = tf.norm(target_item_emb-item_emb, ord='euclidean', axis=1)
        metric_loss = self.label * tf.square(tf.maximum(0., margin - self.distance)) + (1 - self.label) * tf.square(self.distance)
        self.metric_loss = 0.5 * tf.reduce_mean(metric_loss)

        # Compute the MSE Loss for the Record-Level DQNs
        with tf.variable_scope('record_dqn', reuse=tf.AUTO_REUSE):
            q_value = self.primary_record_dqn.forward(state, transition)
            next_q_value = self.target_record_dqn.forward(next_state, next_transition)
            predict_q_value = self.label + gamma*next_q_value
            self.dqn_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_q_value, predictions=q_value))

        # Compute the MSE Loss for the Session-Level DQNs
        with tf.variable_scope('session_dqn', reuse=tf.AUTO_REUSE):
            meta_q_value = self.primary_session_dqn.forward(state, session_policy)
            next_meta_q_value = self.target_session_dqn.forward(next_state, session_policy)
            predict_meta_q_value = self.label + gamma*next_meta_q_value
            self.meta_dqn_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_meta_q_value, predictions=meta_q_value))

        # Aggregation Loss Functions for Backpropogation 
        self.loss = self.metric_loss + self.dqn_loss + self.meta_dqn_loss

        # Step VariableS for Updating Parameters in the Training Process
        self.embedding = target_item_emb
        self.item_emb_table = item_emb_table
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.last_i: uij[2],
                self.hist: uij[3],
                self.next_hist: uij[4],
                self.label: uij[5],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        distance, embedding = sess.run([self.distance, self.embedding], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.last_i: uij[2],
                self.hist: uij[3],
                self.next_hist: uij[4],
                self.label: uij[5]
                })
        return distance, uij[5], uij[0], uij[1], embedding

    def get_item_emb(self, sess, i_table):
        item_emb_table = sess.run(self.item_emb_table, feed_dict={
                self.i_table: i_table
                })
        return item_emb_table

    def seq_attention(self, inputs, hidden_size, attention_size):
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas

class DQN(object):
    def __init__(self, hidden_size, scope_name):
        self.hidden_size = hidden_size
        self.scope_name = scope_name

    def forward(self, state, transition):
        with tf.variable_scope(self.scope_name):
            concat = tf.concat([state, transition], axis=1)
            concat = tf.layers.dense(concat, self.hidden_size, activation=tf.nn.relu)
            q_value = tf.layers.dense(concat, 1, activation=tf.nn.tanh)
            q_value = tf.reshape(q_value, [-1])
            return q_value