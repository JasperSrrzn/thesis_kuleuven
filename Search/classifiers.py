import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from datetime import datetime
#import dill

def accuracy(predictions,labels):
    """
    Accuracy of a given set of predictions of size (N x n_classes) and
    labels of size (N x n_classes)
    """
    return  np.sum(np.argmax(predictions,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

def construct_architecture(X,dropout_rate, layer_ids, layer_dims):
    """
    function to construct network architecture
    X is input (DataFrame or numpy)
    dropout_rate is dropout rate (int)
    layer_ids is list of layer names
    layer_dims is list of layer dimensions
    """
    for idx, lid in enumerate(layer_ids):
        with tf.variable_scope(lid):
            w = tf.get_variable('weights',shape=[layer_dims[idx], layer_dims[idx+1]],initializer=tf.truncated_normal_initializer(stddev=0.05))
            b = tf.get_variable('bias',shape=[layer_dims[idx+1]],initializer = tf.random_uniform_initializer(-0.1,0.1))


    h = X
    for lid in layer_ids:
        with tf.variable_scope(lid,reuse=True):
            w, b = tf.get_variable('weights'), tf.get_variable('bias')
            if lid != 'out':
                h = tf.nn.relu(tf.matmul(h,w)+b,name=lid+'_output')
            else:
                h = tf.nn.xw_plus_b(h,w,b,name=lid+'_ouput')

    all_summaries = []
    for lid in layer_ids:
        with tf.name_scope(lid+'_hist'):
            with tf.variable_scope(lid,reuse=True):
                w,b = tf.get_variable('weights'), tf.get_variable('bias')
                tf_w_hist = tf.summary.histogram('weights_hist',tf.reshape(w,[-1]))
                tf_b_hist = tf.summary.histogram('bias_hist',b)
                all_summaries.extend([tf_w_hist,tf_b_hist])
    tf_param_summaries = tf.summary.merge(all_summaries)


    return h, tf_param_summaries



class SimpleClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,batch_size=128,n_input=200,dropout_rate=0.2,learning_rate = 1e-3, n_epochs = 20,
                    layer_ids=['hidden1','hidden2','hidden3','out']):
        self.batch_size = batch_size
        self.n_input = n_input
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.layer_ids = layer_ids


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self,d):
        self.__dict__ = d
        return self


    def fit(self,x,y,x_validation=None,y_validation=None):
        n_input= x.shape[-1]
        embed_model = 'transe'
        dimension = 50
        X = tf.placeholder("float",[None,n_input],name="input_features")
        Y = tf.placeholder("float",[None,2],name="input_labels")
        layer_dimensions = [n_input,int(0.75*n_input),(0.5*n_input),(0.25*n_input),2]
        logits, tf_param_summaries = construct_architecture(X,self.dropout_rate,self.layer_ids,layer_dimensions)

        tf_predictions = tf.nn.softmax(logits,name='predictions')
        tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits),name='loss')

        tf_learning_rate = tf.placeholder(tf.float32,shape=None, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(tf_learning_rate)
        grad_and_vars = optimizer.compute_gradients(tf_loss)
        tf_loss_minimize = optimizer.minimize(tf_loss)

        with tf.name_scope('performance'):
            tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
            tf_loss_summary = tf.summary.scalar('loss',tf_loss_ph)
            tf_accuracy_ph = tf.placeholder(tf.float32,shape=None,name='accuracy_summary')
            tf_accuracy_summary = tf.summary.scalar('accuracy',tf_accuracy_ph)
            tf_auc_ph = tf.placeholder(tf.float32,shape=None,name='auc_summary')
            tf_auc_summary = tf.summary.scalar('auc',tf_auc_ph)

        for g, v in grad_and_vars:
            if 'hidden3' in v.name and 'weights' in v.name:
                with tf.name_scope('gradients'):
                    tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                    tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)

        performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary, tf_auc_summary])

        date = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
        train_summary_writer = tf.summary.FileWriter('./logs/'+str(embed_model)+'_'+str(dimension)+'/'+str(date)+'/train',tf.Session().graph)
        validation_summary_writer = tf.summary.FileWriter('./logs/'+str(embed_model)+'_'+str(dimension)+'/'+str(date)+'/validation')

        saver = tf.train.Saver(tf.all_variables(),max_to_keep=1)

        with tf.Session() as sess:
            init = tf.global_variables_initializer(), tf.local_variables_initializer()
            sess.run(init)
            for epoch in range(self.n_epochs):
                train_loss_per_epoch = []
                total_batch_train = int(len(x)/self.batch_size)
                x_batches_train = np.array_split(x,total_batch_train)
                y_batches_train = np.array_split(y,total_batch_train)
                for i in range(total_batch_train):
                    batch_x_train = x_batches_train[i]
                    batch_y_train = y_batches_train[i]
                    if i == 0:
                        l,_,gn_summ, wb_summ = sess.run([tf_loss,tf_loss_minimize,tf_gradnorm_summary,tf_param_summaries],feed_dict={X:batch_x_train, Y:batch_y_train, tf_learning_rate:0.0001})
                        train_summary_writer.add_summary(gn_summ,epoch)
                        train_summary_writer.add_summary(wb_summ,epoch)
                    else:
                        l,_ = sess.run([tf_loss,tf_loss_minimize],feed_dict={X:batch_x_train,Y:batch_y_train,tf_learning_rate:0.0001})
                    train_loss_per_epoch.append(l)
                avg_train_loss = np.mean(train_loss_per_epoch)

                print('\tAverage loss in epoch %d: %.5f'%(epoch,avg_train_loss))

                train_prediction = sess.run(tf_predictions,feed_dict={X:x})
                avg_train_acc = accuracy(train_prediction,y)
                avg_train_auc = roc_auc_score(y,train_prediction)

                summ_train = sess.run(performance_summaries,feed_dict={tf_loss_ph:avg_train_loss, tf_accuracy_ph:avg_train_acc, tf_auc_ph:avg_train_auc})
                train_summary_writer.add_summary(summ_train,epoch)
                train_summary_writer.flush

                if y_validation!=None and x_validation!=None:
                    valid_loss_per_epoch = []
                    total_batch_valid = int(len(x_validation)/self.batch_size)
                    x_batches_valid = np.array_split(x_validation,total_batch_valid)
                    y_batches_valid = np.array_split(y_validation,total_batch_valid)
                    for i in range(total_batch_valid):
                        batch_x_valid = x_batches_valid[i]
                        batch_y_valid = y_batches_valid[i]
                        valid_batch_loss = sess.run(tf_loss,feed_dict={X:batch_x_valid,Y:batch_y_valid})
                        valid_loss_per_epoch.append(valid_batch_loss)
                    avg_valid_loss = np.mean(valid_loss_per_epoch)

                    valid_prediction = sess.run(tf_predictions,feed_dict={X:x_validation})
                    avg_valid_acc = accuracy(valid_prediction,y_validation)
                    avg_valid_auc = roc_auc_score(y_validation,valid_prediction)
                    summ_valid = sess.run(performance_summaries,feed_dict={tf_loss_ph:avg_valid_loss, tf_accuracy_ph:avg_valid_acc, tf_auc_ph:avg_valid_auc})
                    validation_summary_writer.add_summary(summ_valid,epoch)
                    validation_summary_writer.flush()


                print('--------------------------------------------')

                saver.save(sess, "./classification_models/model_"+embed_model+'_'+str(dimension)+'/',global_step=epoch)

        return self

    def predict_proba(self,filepath,x,y=None,):
        tf.logging.set_verbosity(tf.logging.ERROR)
        with tf.Session() as sess:
            n_epochs = self.n_epochs
            tf_saver=tf.train.import_meta_graph(filepath+'-'+str(n_epochs-1)+'.meta')
            tf_saver.restore(sess,tf.train.latest_checkpoint(filepath))
            return sess.run('predictions:0',feed_dict={'input_features:0':x})
