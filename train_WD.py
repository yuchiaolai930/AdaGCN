import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from utils import *
from models import GCN

# Set random seed
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source', 'citationv1', 'Source dataset string.')
flags.DEFINE_string('target', 'dblpv7', 'Target dataset string.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('l2_param', 5e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('signal', 4, 'The network to train: 1-with domain adaptation (source_only), 2-with domain adaptation (source and target).')
flags.DEFINE_float('da_param', 1, 'Weight for wasserstein loss.')
flags.DEFINE_float('gp_param', 10, 'Weight for penalty loss.')
flags.DEFINE_integer('D_train_step', 10, 'The number of steps for training discriminator.')
flags.DEFINE_float('shrinking', 0.8, 'Initial learning rate for discriminator.')
flags.DEFINE_float('train_rate', 0, 'The ratio of labeled nodes in target networks.')
flags.DEFINE_float('val_rate', 0, 'The ratio of labeled nodes in validation set in target networks.')
flags.DEFINE_string('hiddens_gcn', '1000|100|16', 'Number of units in different hidden layers for gcn.')
flags.DEFINE_string('hiddens_clf', '', 'Number of units in different hidden layers for supervised classifier.')
flags.DEFINE_string('hiddens_dis', '16', 'Number of units in different hidden layers for discriminator.')
flags.DEFINE_string('da_method', 'WD', 'Domain adaptation method.')
flags.DEFINE_float('lr_gen', 1.5e-3, 'Initial learning rate.')
flags.DEFINE_float('lr_dis', 1.5e-3, 'Initial learning rate for discriminator.')
flags.DEFINE_boolean('with_metrics', True, 'whether computing f1 scores within tensorflow.')
flags.DEFINE_float('source_train_rate', 0.1, 'The ratio of labeled nodes in target networks.')
flags.DEFINE_integer('num_gcn_layers', 1, 'The number of gcn layers in the IGCN model.')
flags.DEFINE_integer('smoothing_steps', 10, 'The setting of k in A^k.')
flags.DEFINE_string('gnn', 'igcn', 'Convolutional methods.')

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Ensure hidden1 is defined
hidden1 = int(FLAGS.hiddens_gcn.split('|')[0])

# Load data
datasets = ['dblpv7', 'citationv1', 'acmv9']
signals = [1]
target_train_rate = [0]
smoothing_steps = [10]
lr_gen = 1.5e-3
lr_dis = 1.5e-3

def evaluate(sess, model, features_t, labels_t, support_t, mask_t, features_s, labels_s, support_s, mask_s, placeholders, lr_gen, lr_dis):
    """Evaluate model performance."""
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features_t, labels_t, support_t, mask_t,
        features_s, labels_s, support_s, mask_s,
        placeholders, lr_gen, lr_dis
    )
    feed_dict_val.update({placeholders['dropout']: 0.0})
    
    outs = sess.run([model.clf_loss, model.micro_f1_t, model.macro_f1_t, model.weighted_f1_t], feed_dict=feed_dict_val)
    return outs[0], outs[1], outs[2], outs[3], (time.time() - t_test)

def construct_feed_dict(features_t, labels_t, support_t, labels_mask_t, features_s, labels_s, support_s, labels_mask_s, placeholders, lr_gen, lr_dis):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['support_t'][i]: support_t[i] for i in range(len(support_t))})
    feed_dict.update({placeholders['features_t']: features_t})
    feed_dict.update({placeholders['labels_t']: labels_t})
    feed_dict.update({placeholders['labels_mask_t']: labels_mask_t})
    feed_dict.update({placeholders['num_features_nonzero_t']: features_t[1].shape})
    
    feed_dict.update({placeholders['support_s'][i]: support_s[i] for i in range(len(support_s))})
    feed_dict.update({placeholders['features_s']: features_s})
    feed_dict.update({placeholders['labels_s']: labels_s})
    feed_dict.update({placeholders['labels_mask_s']: labels_mask_s})
    feed_dict.update({placeholders['num_features_nonzero_s']: features_s[1].shape})
    
    feed_dict.update({placeholders['lr_gen']: lr_gen})
    feed_dict.update({placeholders['lr_dis']: lr_dis})
    
    return feed_dict

for s in range(len(signals)):
    FLAGS.signal = signals[s]
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i == j:
                continue
            FLAGS.source = datasets[j]
            FLAGS.target = datasets[i]
            
            final_micro = []
            final_macro = []
            for k, t in enumerate(target_train_rate):
                FLAGS.lr_gen = lr_gen
                FLAGS.lr_dis = lr_dis
                FLAGS.train_rate = t
                FLAGS.smoothing_steps = smoothing_steps[k]

                # Load target data
                Y_tmp = sio.loadmat(f'./input/{FLAGS.target}.mat')['group']
                N_tmp, M_tmp = Y_tmp.shape
                Ntr_s = int(N_tmp * FLAGS.train_rate)
                tr_label_per_class = Ntr_s // M_tmp
                label_node_min = np.min(np.sum(Y_tmp, axis=0))

                if tr_label_per_class > label_node_min:
                    s_type = 'random'
                else:
                    s_type = 'planetoid'
                
                train_ratio = FLAGS.train_rate
                val_ratio = FLAGS.val_rate
                test_ratio = 1 - FLAGS.train_rate - FLAGS.val_rate
                
                A_t, X_t, Y_t, y_train_t, y_val, y_test, train_mask_t, val_mask, test_mask = load_mat_data(
                    f'./input/{FLAGS.target}.mat',
                    train_ratio, val_ratio, test_ratio, s_type=s_type
                )
                
                # Load source data
                Y_tmp = sio.loadmat(f'./input/{FLAGS.source}.mat')['group']
                N_tmp, M_tmp = Y_tmp.shape
                Ntr_s = int(N_tmp * FLAGS.source_train_rate)
                tr_label_per_class = Ntr_s // M_tmp
                label_node_min = np.min(np.sum(Y_tmp, axis=0))

                if tr_label_per_class > label_node_min:
                    s_type = 'random'
                else:
                    s_type = 'planetoid'
                
                source_train_ratio = FLAGS.source_train_rate
                source_val_ratio = 0
                source_test_ratio = 1 - source_train_ratio - source_val_ratio
                
                A_s, X_s, Y_s, y_train_s, y_val_s, y_test_s, train_mask_s, val_mask_s, test_mask_s = load_mat_data(
                    f'./input/{FLAGS.source}.mat',
                    source_train_ratio, source_val_ratio, source_test_ratio, s_type=s_type
                )
                
                                # Some preprocessing
                N_t = Y_t.shape[0]
                N_s = Y_s.shape[0]
                X_t = preprocess_features(X_t)
                X_s = preprocess_features(X_s)
                support_t = [preprocess_adj(A_t)]
                support_s = [preprocess_adj(A_s)]
                num_supports = 1
                
                # Define placeholders
                placeholders = {
                    'support_t': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    'features_t': tf.compat.v1.sparse_placeholder(tf.float32),
                    'labels_t': tf.compat.v1.placeholder(tf.float32, shape=(None, Y_t.shape[1])),
                    'labels_mask_t': tf.compat.v1.placeholder(tf.int32),
                    'num_features_nonzero_t': tf.compat.v1.placeholder(tf.int32),  # helper variable for sparse dropout
                    'support_s': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    'features_s': tf.compat.v1.sparse_placeholder(tf.float32),
                    'labels_s': tf.compat.v1.placeholder(tf.float32, shape=(None, Y_s.shape[1])),
                    'labels_mask_s': tf.compat.v1.placeholder(tf.int32),
                    'num_features_nonzero_s': tf.compat.v1.placeholder(tf.int32),  # helper variable for sparse dropout
                    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
                    'lr_dis': tf.compat.v1.placeholder(tf.float32, shape=()),
                    'lr_gen': tf.compat.v1.placeholder(tf.float32, shape=()),
                    'l': tf.compat.v1.placeholder(tf.int32),  # for selecting which portion to apply adversarial loss
                    'source_top_k_list': tf.compat.v1.placeholder(tf.int32, shape=(None, hidden1)),  # for multi-label classification
                    'target_top_k_list': tf.compat.v1.placeholder(tf.int32, shape=(None, hidden1))
                }

                # Create model
                model = GCN(
                    placeholders, 
                    X_t[2][1], 
                    Y_s.shape[0], 
                    Y_t.shape[0], 
                    FLAGS=FLAGS,  # Ensure this is passed to the model
                    da_method=FLAGS.da_method,
                    epochs=FLAGS.epochs,
                    dropout=FLAGS.dropout,
                    l2_param=FLAGS.l2_param,
                    signal=FLAGS.signal,
                    da_param=FLAGS.da_param,
                    gp_param=FLAGS.gp_param,
                    D_train_step=FLAGS.D_train_step,
                    shrinking=FLAGS.shrinking,
                    train_rate=FLAGS.train_rate,
                    val_rate=FLAGS.val_rate,
                    hiddens_gcn=FLAGS.hiddens_gcn,
                    hiddens_clf=FLAGS.hiddens_clf,
                    hiddens_dis=FLAGS.hiddens_dis,
                    lr_gen=FLAGS.lr_gen,
                    lr_dis=FLAGS.lr_dis,
                    with_metrics=FLAGS.with_metrics,
                    source_train_rate=FLAGS.source_train_rate,
                    num_gcn_layers=FLAGS.num_gcn_layers,
                    smoothing_steps=FLAGS.smoothing_steps,
                    gnn=FLAGS.gnn,  # Ensure this is passed to the model
                    logging=True
                )

                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.compat.v1.Session(config=config)
                sess.run(tf.compat.v1.global_variables_initializer())

                micro_f1 = []
                macro_f1 = []

                # Train model
                for epoch in range(FLAGS.epochs):
                    # Construct feed dictionary
                    if FLAGS.da_method == 'WD':
                        if (epoch + 1) >= 500 and (epoch + 1) % 100 == 0:
                            FLAGS.lr_gen *= FLAGS.shrinking
                            FLAGS.lr_dis *= FLAGS.shrinking

                    feed_dict = construct_feed_dict(
                        X_t, Y_t, support_t, y_train_t, train_mask_t,
                        X_s, Y_s, support_s, y_train_s, train_mask_s, placeholders,
                        FLAGS.lr_gen, FLAGS.lr_dis
                    )
                    
                                        # Ensure that the data fed into the placeholder matches its expected shape
                    source_top_k_list_data = np.random.randint(0, 2, size=(N_s, hidden1))  # Replace this with actual data if available
                    target_top_k_list_data = np.random.randint(0, 2, size=(N_t, hidden1))  # Replace this with actual data if available
                    
                    feed_dict[placeholders['source_top_k_list']] = source_top_k_list_data
                    feed_dict[placeholders['target_top_k_list']] = target_top_k_list_data

                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    
                    if FLAGS.signal == 1:
                        if FLAGS.da_method == 'WD':
                            wd_loss, dis_loss_total = [], []
                            for _ in range(FLAGS.D_train_step):
                                outs_dis = sess.run([model.wd_d_op, model.wd_loss, model.dis_loss_total], feed_dict=feed_dict)
                                wd_loss.append(outs_dis[1])
                                dis_loss_total.append(outs_dis[2])
                            outs_gen = sess.run([model.opt_op_total_s], feed_dict=feed_dict)

                    elif FLAGS.signal == 2:
                        if FLAGS.da_method == 'WD':
                            for _ in range(FLAGS.D_train_step):
                                outs_dis = sess.run([model.wd_d_op, model.wd_loss, model.dis_loss_total], feed_dict=feed_dict)
                            outs_gen = sess.run([model.opt_op_total_s_t], feed_dict=feed_dict)

                    # Recording test results after each epoch
                    test_clf_loss_t, test_micf1, test_macf1, test_wf1, test_duration = evaluate(
                        sess, model, X_t, Y_t, support_t, test_mask,
                        X_s, Y_s, support_s, train_mask_s, placeholders,
                        FLAGS.lr_gen, FLAGS.lr_dis
                    )
                    print(f"Epoch:{epoch+1} signal={FLAGS.signal} S-T:{FLAGS.source}-{FLAGS.target} hiddens={FLAGS.hiddens_gcn} "
                          f"dropout={FLAGS.dropout} l2_param={FLAGS.l2_param} cost={test_clf_loss_t:.3f} "
                          f"micro_f1={test_micf1:.3f} macro_f1={test_macf1:.3f} weighted_f1={test_wf1:.3f}")

                    micro_f1.append(test_micf1)
                    macro_f1.append(test_macf1)

                # Testing
                test_clf_loss_t, test_micf1, test_macf1, test_wf1, test_duration = evaluate(
                    sess, model, X_t, Y_t, support_t, test_mask,
                    X_s, Y_s, support_s, train_mask_s, placeholders,
                    FLAGS.lr_gen, FLAGS.lr_dis
                )
                print(f"signal={FLAGS.signal} S-T:{FLAGS.source}-{FLAGS.target} hiddens={FLAGS.hiddens_gcn} "
                    f"dropout={FLAGS.dropout} l2_param={FLAGS.l2_param} cost={test_clf_loss_t:.3f} "
                    f"micro_f1={test_micf1:.3f} macro_f1={test_macf1:.3f} weighted_f1={test_wf1:.3f}")

                final_micro.append(test_micf1)
                final_macro.append(test_macf1)

                # Final results
                print(f"Final results for signal={FLAGS.signal} S-T:{FLAGS.source}-{FLAGS.target}")
                print(f"Micro-F1: {np.mean(final_micro):.3f} ± {np.std(final_micro):.3f}")
                print(f"Macro-F1: {np.mean(final_macro):.3f} ± {np.std(final_macro):.3f}")
