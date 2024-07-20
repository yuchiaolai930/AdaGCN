import tensorflow as tf
import numpy as np
from utils import *
from layers import GraphConvolution, ImprovedGraphConvolution

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def define_variables(hiddens, weight_name, bias_name, flag=False):
    variables = {}
    for i in range(len(hiddens)-1):
        variables[weight_name.format(i)] = glorot([hiddens[i], hiddens[i+1]], name=weight_name.format(i))
        if flag:
            variables[bias_name.format(i)] = zeros([hiddens[i+1]], name=bias_name.format(i))
    return variables

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Sigmoid cross-entropy loss with masking."""
    # Ensure shapes match
    labels = tf.reshape(labels, [-1, preds.get_shape().as_list()[-1]])
    mask = tf.reshape(mask, [-1])

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask[:, None]  # Ensure mask shape matches loss shape
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def multi_label_hot(preds, threshold=0.5):
    """Convert predictions to multi-label hot encoding."""
    return tf.cast(preds > threshold, dtype=tf.float32)

def f1_score(preds, labels, mask):
    """Calculate F1 score."""
    preds = tf.cast(preds, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    TP = tf.reduce_sum(tf.cast(preds * labels, dtype=tf.float32), axis=0)
    FP = tf.reduce_sum(tf.cast(preds * (1 - labels), dtype=tf.float32), axis=0)
    FN = tf.reduce_sum(tf.cast((1 - preds) * labels, dtype=tf.float32), axis=0)

    precision = TP / (TP + FP + tf.keras.backend.epsilon())
    recall = TP / (TP + FN + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    micro_f1 = tf.reduce_mean(f1)
    macro_f1 = tf.reduce_mean(f1, axis=-1)
    weighted_f1 = tf.reduce_sum(f1 * mask) / tf.reduce_sum(mask)

    return micro_f1, macro_f1, weighted_f1, TP, FP, FN

class GCN(object):
    def __init__(self, placeholders, input_dim, N_s, N_t, FLAGS, bias_flag=False, c_type='multi-label', **kwargs):
        allowed_kwargs = {
            'name', 'logging', 'da_method', 'epochs', 'dropout', 'l2_param', 'signal', 'da_param', 
            'gp_param', 'D_train_step', 'shrinking', 'train_rate', 'val_rate', 'hiddens_gcn', 
            'hiddens_clf', 'hiddens_dis', 'lr_gen', 'lr_dis', 'with_metrics', 'source_train_rate', 
            'num_gcn_layers', 'smoothing_steps', 'gnn'
        }
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.da_method = kwargs.get('da_method', 'WD')
        self.epochs = kwargs.get('epochs', 1000)
        self.dropout = kwargs.get('dropout', 0.3)
        self.l2_param = kwargs.get('l2_param', 5e-5)
        self.signal = kwargs.get('signal', 4)
        self.da_param = kwargs.get('da_param', 1)
        self.gp_param = kwargs.get('gp_param', 10)
        self.D_train_step = kwargs.get('D_train_step', 10)
        self.shrinking = kwargs.get('shrinking', 0.8)
        self.train_rate = kwargs.get('train_rate', 0)
        self.val_rate = kwargs.get('val_rate', 0)
        self.hiddens_gcn = kwargs.get('hiddens_gcn', '1000|100|16')
        self.hiddens_clf = kwargs.get('hiddens_clf', '')
        self.hiddens_dis = kwargs.get('hiddens_dis', '16')
        self.lr_gen = kwargs.get('lr_gen', 1.5e-3)
        self.lr_dis = kwargs.get('lr_dis', 1.5e-3)
        self.with_metrics = kwargs.get('with_metrics', True)
        self.source_train_rate = kwargs.get('source_train_rate', 0.1)
        self.num_gcn_layers = kwargs.get('num_gcn_layers', 1)
        self.smoothing_steps = kwargs.get('smoothing_steps', 10)
        self.gnn = kwargs.get('gnn', 'igcn')
        
        self.bias_flag = bias_flag
        self.c_type = c_type
        self.FLAGS = FLAGS

        self.inputs_t = placeholders['features_t']
        self.inputs_s = placeholders['features_s']
        self.output_dim = placeholders['labels_t'].shape[1]

        self.N = N_s if N_s < N_t else N_t
        self.N_s = N_s
        self.N_t = N_t

        self.hiddens_gcn = [input_dim] + [int(h) for h in self.hiddens_gcn.split('|')]
        self.hiddens_clf = [self.hiddens_gcn[-1]] + [int(h) for h in self.hiddens_clf.split('|') if h != ''] + [self.output_dim]
        self.hiddens_dis = [self.hiddens_gcn[-1]] + [int(h) for h in self.hiddens_dis.split('|') if h != ''] + [1]

        self.vars = {}

        self.layers_t = []
        self.activations_t = []
        self.layers_s = []
        self.activations_s = []

        self.clf_outputs_t = None
        self.hiddens_t = None

        self.clf_outputs_s = None
        self.hiddens_s = None

        self.clf_loss = 0
        self.clf_loss_t = 0
        self.clf_loss_s = 0
        self.opt_op = None
        self.opt_op_t = None
        self.opt_op_s = None
        self.placeholders = placeholders
        self._create_variables()  # Ensure variables are created before build
        self.build()

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        theta_C = [v for v in tf.compat.v1.global_variables() if 'clf' in v.name]
        theta_D = [v for v in tf.compat.v1.global_variables() if 'dis' in v.name]
        theta_G = [v for v in tf.compat.v1.global_variables() if 'gcn' in v.name]

        ##############
        # Generator
        ##############
        # Build sequential layer model
        self.activations_t.append(self.inputs_t)
        for layer in self.layers_t:
            hidden = layer(self.activations_t[-1])
            self.activations_t.append(hidden)
        self.hiddens_t = self.activations_t[-1]

        self.activations_s.append(self.inputs_s)
        for layer in self.layers_s:
            hidden = layer(self.activations_s[-1])
            self.activations_s.append(hidden)
        self.hiddens_s = self.activations_s[-1]

        ########################
        # Classifier
        ########################
        self.clf_outputs_t = self._classifier(self.hiddens_t)
        self.clf_outputs_s = self._classifier(self.hiddens_s)

        ########################
        # Discriminator
        ########################
        if self.da_method == 'WD':
            # generate samples for gradient penalty term
            #-----------------------------------------------------------------------------------
            if self.N_s < self.N_t:
                hiddens_s = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_t_1 = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])
                hiddens_t_2 = tf.slice(self.hiddens_t, [self.N_t - self.N - 1, 0], [self.N, -1])

                hiddens_s = tf.concat([hiddens_s, hiddens_s], axis=0)
                hiddens_t = tf.concat([hiddens_t_1, hiddens_t_2], axis=0)

                alpha = tf.random.uniform(shape=[2 * self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)
            elif self.N_s > self.N_t:
                hiddens_s_1 = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_s_2 = tf.slice(self.hiddens_s, [self.N_s - self.N - 1, 0], [self.N, -1])
                hiddens_t = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])

                hiddens_s = tf.concat([hiddens_s_1, hiddens_s_2], axis=0)
                hiddens_t = tf.concat([hiddens_t, hiddens_t], axis=0)

                alpha = tf.random.uniform(shape=[2 * self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)
            else:
                hiddens_s = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_t = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])
                alpha = tf.random.uniform(shape=[self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)
            #-----------------------------------------------------------------------------------

            hiddens_whole = tf.concat([self.hiddens_s, self.hiddens_t, interpolates], axis=0)

            # critic loss
            critic_out = self._discriminator(hiddens_whole)
            critic_s = tf.slice(critic_out, [0, 0], [self.N_s, -1])
            critic_t = tf.slice(critic_out, [self.N_s, 0], [self.N_t, -1])
            self.wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))

            # gradient penalty
            gradients = tf.gradients(critic_out, [hiddens_whole])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))  # Updated line
            self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            # optimizer
            self.dis_loss_total = -self.wd_loss + self.FLAGS.gp_param * self.gradient_penalty
            self.wd_d_op = tf.compat.v1.train.AdamOptimizer(self.placeholders['lr_dis']).minimize(self.dis_loss_total, var_list=theta_D)
        else:
            raise RuntimeError('Wrong DA Type!')

        ####################
        # Weight decay loss
        ####################
        self.l2_loss = self.FLAGS.l2_param * tf.add_n([tf.nn.l2_loss(v) for v in self.vars.values() if ('bias' not in v.name and 'dis' not in v.name)])

        ###################
        # supervised loss
        ###################
        if self.c_type == 'multi-label':
            self.clf_loss_t = self.l2_loss + masked_sigmoid_cross_entropy(self.clf_outputs_t, self.placeholders['labels_t'],
                                                                          self.placeholders['labels_mask_t'])
            self.clf_loss_s = self.l2_loss + masked_sigmoid_cross_entropy(self.clf_outputs_s, self.placeholders['labels_s'],
                                                                          self.placeholders['labels_mask_s'])
            self.clf_loss = self.clf_loss_t + self.clf_loss_s - self.l2_loss
            self.clf_loss_pure = self.clf_loss - self.l2_loss
        elif self.c_type == 'single-label':
            self.clf_loss_t = self.l2_loss + masked_softmax_cross_entropy(self.clf_outputs_t, self.placeholders['labels_t'],
                                                                          self.placeholders['labels_mask_t'])
            self.clf_loss_s = self.l2_loss + masked_softmax_cross_entropy(self.clf_outputs_s, self.placeholders['labels_s'],
                                                                          self.placeholders['labels_mask_s'])
            self.clf_loss = self.clf_loss_t + self.clf_loss_s - self.l2_loss
            self.clf_loss_pure = self.clf_loss - self.l2_loss

        ###################
        # total loss
        ###################
        if self.da_method == 'WD':
            self.total_loss_s = self.clf_loss_s + self.FLAGS.da_param * self.wd_loss
            self.total_loss_s_t = self.clf_loss + self.FLAGS.da_param * self.wd_loss
        else:
            raise RuntimeError('Wrong DA Type!')

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.placeholders['lr_gen'])
        self.opt_op_total_s = self.optimizer.minimize(self.total_loss_s, var_list=theta_G + theta_C)
        self.opt_op_total_s_t = self.optimizer.minimize(self.total_loss_s_t, var_list=theta_G + theta_C)

        # Build metrics
        if self.FLAGS.with_metrics:
            # target
            predictions_t = multi_label_hot(tf.sigmoid(self.clf_outputs_t))
            self.micro_f1_t, self.macro_f1_t, self.weighted_f1_t, self.TP_t, self.FP_t, self.FN_t = f1_score(predictions_t,
                                                                                                              self.placeholders['labels_t'],
                                                                                                              self.placeholders['labels_mask_t'])
            # source
            predictions_s = multi_label_hot(tf.sigmoid(self.clf_outputs_s))
            self.micro_f1_s, self.macro_f1_s, self.weighted_f1_s, self.TP_s, self.FP_s, self.FN_s = f1_score(predictions_s,
                                                                                                              self.placeholders['labels_s'],
                                                                                                              self.placeholders['labels_mask_s'])

    def _classifier(self, input_tensor, act=tf.nn.relu):
        """classification"""
        hiddens = input_tensor
        for i in range(len(self.hiddens_clf) - 2):
            hiddens = tf.nn.dropout(hiddens, 1 - self.placeholders['dropout'])
            hiddens = tf.add(tf.matmul(hiddens, self.vars['clf_{}_weights'.format(i)]), self.vars['clf_{}_bias'.format(i)])
            hiddens = act(hiddens)
        output = tf.add(tf.matmul(hiddens, self.vars['clf_{}_weights'.format(len(self.hiddens_clf) - 2)]),
                        self.vars['clf_{}_bias'.format(len(self.hiddens_clf) - 2)])
        return output

    def _discriminator(self, input_tensor, act=tf.nn.tanh):
        """discriminator"""
        hiddens = input_tensor
        for i in range(len(self.hiddens_dis) - 2):
            hiddens = tf.add(tf.matmul(hiddens, self.vars['dis_{}_weights'.format(i)]), self.vars['dis_{}_bias'.format(i)])
            hiddens = act(hiddens)
        output = tf.add(tf.matmul(hiddens, self.vars['dis_{}_weights'.format(len(self.hiddens_dis) - 2)]),
                        self.vars['dis_{}_bias'.format(len(self.hiddens_dis) - 2)])
        return output

    def _build(self):
        """ Wrapper for _create_generator() and _create_classifier() """
        self._create_generator()
        self._create_classifier()

    def _create_generator(self):
        """Create Generator"""
        # define model layers for generator
        for i in range(len(self.hiddens_gcn) - 1):
            if i == 0:
                sparse = True
                act = tf.nn.relu
            else:
                sparse = False
                act = tf.nn.relu

            if self.bias_flag:
                bias = self.vars['gcn_{}_bias'.format(i)]
            else:
                bias = None

            if self.gnn == 'gcn':
                self.layers_t.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                      output_dim=self.hiddens_gcn[i + 1],
                                                      placeholder_dropout=self.placeholders['dropout'],
                                                      placeholder_support=self.placeholders['support_t'],
                                                      placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_t'],
                                                      weights=self.vars['gcn_{}_weights'.format(i)],
                                                      bias=bias,
                                                      act=act,
                                                      sparse_inputs=sparse))
                self.layers_s.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                      output_dim=self.hiddens_gcn[i + 1],
                                                      placeholder_dropout=self.placeholders['dropout'],
                                                      placeholder_support=self.placeholders['support_s'],
                                                      placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_s'],
                                                      weights=self.vars['gcn_{}_weights'.format(i)],
                                                      bias=bias,
                                                      act=act,
                                                      sparse_inputs=sparse))
            elif self.gnn == 'igcn':
                self.layers_t.append(ImprovedGraphConvolution(input_dim=self.hiddens_gcn[i],
                                                              output_dim=self.hiddens_gcn[i + 1],
                                                              placeholder_dropout=self.placeholders['dropout'],
                                                              placeholder_support=self.placeholders['support_t'],
                                                              placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_t'],
                                                              weights=self.vars['gcn_{}_weights'.format(i)],
                                                              bias=bias,
                                                              act=act,
                                                              sparse_inputs=sparse))
                self.layers_s.append(ImprovedGraphConvolution(input_dim=self.hiddens_gcn[i],
                                                              output_dim=self.hiddens_gcn[i + 1],
                                                              placeholder_dropout=self.placeholders['dropout'],
                                                              placeholder_support=self.placeholders['support_s'],
                                                              placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_s'],
                                                              weights=self.vars['gcn_{}_weights'.format(i)],
                                                              bias=bias,
                                                              act=act,
                                                              sparse_inputs=sparse))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn}")

    def _create_classifier(self):
        """Create Classifier"""
        for i in range(len(self.hiddens_clf) - 1):
            self.vars['clf_{}_weights'.format(i)] = glorot([self.hiddens_clf[i], self.hiddens_clf[i + 1]], name='clf_{}_weights'.format(i))
            self.vars['clf_{}_bias'.format(i)] = zeros([self.hiddens_clf[i + 1]], name='clf_{}_bias'.format(i))

    def _create_variables(self):
        # define variables
        vars_gcn = define_variables(self.hiddens_gcn, weight_name='gcn_{}_weights', bias_name='gcn_{}_bias', flag=self.bias_flag)
        vars_clf = define_variables(self.hiddens_clf, weight_name='clf_{}_weights', bias_name='clf_{}_bias', flag=True)
        vars_dis = define_variables(self.hiddens_dis, weight_name='dis_{}_weights', bias_name='dis_{}_bias', flag=True)

        self.vars.update(vars_gcn)
        self.vars.update(vars_clf)
        self.vars.update(vars_dis)
