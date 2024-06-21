import sys
import os
import numpy as np
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import time
import datetime
import csv
from random import shuffle
import tensorflow as tf
# import internal scripts
from tools.tools import *
from test import test
from matplotlib import pyplot as plt
plt.ion()
import gc
import psutil
###############################################################################
#@tf.function
def batch_train_step(n_step):
    '''combines multiple  graph inputs and executes a step on their mean'''

    with tf.GradientTape() as tape:

        for batch in range(config['batch_size']):

            X, Ri, Ro, y = train_data[
                train_list[n_step*config['batch_size']+batch]
                ]
            label = tf.reshape(tf.convert_to_tensor(y),shape=(y.shape[0],1))
            
            if batch==0:
                # calculate weight for each edge to avoid class imbalance
                weights = tf.convert_to_tensor(true_fake_weights(y))

                # reshape weights
                weights = tf.reshape(tf.convert_to_tensor(weights),
                                     shape=(weights.shape[0],1))
                #preds = model([map2angle(X),Ri,Ro])
                preds = model([X,Ri,Ro])
                preds_batch = preds
                labels = label
            else:
                weight = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                weight = tf.reshape(tf.convert_to_tensor(weight),
                                    shape=(weight.shape[0],1))
                weights = tf.concat([weights, weight],axis=0)
                #preds_batch = model([map2angle(X),Ri,Ro])
                preds_batch = model([X,Ri,Ro])
                preds = tf.concat([preds, preds_batch],axis=0)
                labels = tf.concat([labels, label],axis=0)

                tf.keras.backend.clear_session()
                gc.collect()

            #np.savez("/home/peter/Software/trd-standalone-tracking/Data/MC/predictions/predictions_batch%i.npy"%batch, X=X, Ri=Ri, Ro=Ro, y=label, preds=preds_batch)
        loss_eval = loss_fn(labels, preds, sample_weight=weights)

    grads = tape.gradient(loss_eval, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    tf.keras.backend.clear_session()
    gc.collect()

    return loss_eval, grads

def valid():
    '''combines multiple  graph inputs and executes a step on their mean'''
    with tf.GradientTape() as tape:
        for file in range(config['n_valid']):
            X, Ri, Ro, y = valid_data[
                valid_list[file]
                ]

            label = tf.reshape(tf.convert_to_tensor(y),shape=(y.shape[0],1))
            
            if file==0:
                # calculate weight for each edge to avoid class imbalance
                weights = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                weights = tf.reshape(tf.convert_to_tensor(weights),
                                     shape=(weights.shape[0],1))
                #preds = model([map2angle(X),Ri,Ro])
                preds = model([X,Ri,Ro])
                preds_batch = preds
                labels = label
            else:
                weight = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                weight = tf.reshape(tf.convert_to_tensor(weight),
                                    shape=(weight.shape[0],1))

                weights = tf.concat([weights, weight],axis=0)
                #preds_batch = model([map2angle(X),Ri,Ro])
                preds_batch = model([X,Ri,Ro])
                preds = tf.concat([preds, preds_batch],axis=0)
                labels = tf.concat([labels, label],axis=0)

            #np.savez("/home/peter/Software/trd-standalone-tracking/Data/MC/predictions/predictions_batch%i.npy"%batch, X=X, Ri=Ri, Ro=Ro, y=label, preds=preds_batch)
        loss_eval = loss_fn(labels, preds, sample_weight=weights)

    return loss_eval

if __name__ == '__main__':
    # Read config file
    config = load_config(parse_args())
    tools.config = config

    # Set GPU variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    USE_GPU = (config['gpu']  != '-1')

    # Set number of thread to be used
    os.environ['OMP_NUM_THREADS'] = str(config['n_thread'])  # set num workers
    tf.config.threading.set_intra_op_parallelism_threads(config['n_thread'])
    tf.config.threading.set_inter_op_parallelism_threads(config['n_thread'])

    # Load the network
    if config['network'] == 'QGNN':
        from qnetworks.QGNN import GNN
        GNN.config = config
    elif config['network'] == 'CGNN':
        from qnetworks.CGNN import GNN
        GNN.config = config
    else: 
        print('Wrong network specification!')
        sys.exit()
	
    # setup model
    model = GNN()

    # load data
    train_data = get_dataset(config['train_dir'], config['n_train'])
    train_list = [i for i in range(config['n_train'])]

    # execute the model on an example data to test things
    X, Ri, Ro, y = train_data[0]
    #model([map2angle(X), Ri, Ro])
    model([X, Ri, Ro])
    # print model summary
    print(model.summary())

    valid_data = get_dataset(config['valid_dir'], config['n_valid'])
    valid_list = [i for i in range(config['n_valid'])]

    # Log initial parameters if new run
    if config['run_type'] == 'new_run':    
        if config['log_verbosity']>=2:
            log_parameters(config['log_dir'], model.trainable_variables)
        epoch_start = 0

        # Test the validation and training set
        if config['n_valid']: test(config, model, 'valid')
        if config['n_train']: test(config, model, 'train')
    # Load old parameters if continuing run
    elif config['run_type'] == 'continue':
        # load params 
        model, epoch_start = load_params(model, config['log_dir'])
    else:
        raise ValueError('Run type not defined!')

    # Get loss function and optimizer
    loss_fn = getattr(tf.keras.losses, config['loss_func'])()
    opt = getattr(
        tf.keras.optimizers,
        config['optimizer'])(learning_rate=config['lr_c']
    )

    # Print final message before training
    if epoch_start == 0: 
        print(str(datetime.datetime.now()) + ': Training is starting!')
    else:
        print(
            str(datetime.datetime.now()) 
            + ': Training is continuing from epoch {}!'.format(epoch_start+1)
            )

    # Start training
    fig, ax = plt.subplots()
    epoch_losses = []
    batch_losses = []
    epoch_losses_valid = []
    
    for epoch in range(epoch_start, config['n_epoch']):

        shuffle(train_list) # shuffle the order every epoch
        batch_losses = []

        for n_step in range(config['n_train']//config['batch_size']):

            # start timer
            t0 = datetime.datetime.now()  

            # iterate a step
            loss_eval, grads = batch_train_step(n_step)
            print(np.shape(loss_eval), np.shape(grads))
            print("Memory usage before clearing: ", psutil.virtual_memory().percent)
            batch_losses.append(loss_eval.numpy())
            print("Memory usage after clearing: ", psutil.virtual_memory().percent)
                        
            # end timer
            dt = datetime.datetime.now() - t0  
            t = dt.seconds + dt.microseconds * 1e-6 # time spent in seconds

            # Print summary
            print(
                str(datetime.datetime.now())
                + ": Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" \
                %(epoch+1, n_step+1, loss_eval.numpy() ,t / 60, t % 60)
                )
            
            # Start logging 
            
            # Log summary 
            with open(config['log_dir']+'summary.csv', 'a') as f:
                f.write(
                    '%d, %d, %f, %f\n' \
                    %(epoch+1, n_step+1, loss_eval.numpy(), t)
                    )

	       # Log parameters
            if config['log_verbosity']>=2:
                log_parameters(config['log_dir'], model.trainable_variables)

           # Log gradients
            if config['log_verbosity']>=2:
                log_gradients(config['log_dir'], grads)
            
            # Test every TEST_every
            if (n_step+1)%config['TEST_every']==0:
                test(config, model, 'valid')
                test(config, model, 'train')
        
        epoch_losses_valid.append(valid())

        if(len(batch_losses)==0):
            epoch_loss = 0
        else:
            epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)
        
        # Plot the epoch losses
        ax.clear()  # Clear the previous plot
        ax.plot(range(1, epoch+2), epoch_losses, label='Epoch Loss Training Set', marker='o', linestyle='-')
        ax.plot(range(1, epoch+2), epoch_losses_valid, label='Epoch Loss Validation Set', marker='o', linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Average Training Loss per Epoch')
        ax.set_ylim(bottom=0)
        ax.set_xlim([0, config['n_epoch']])
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    plt.savefig("./pdf/graphs/losses.pdf")
    plt.ioff()
    for i in range(len(train_data)):   
        X, Ri, Ro, y = train_data[i]
        preds = model([X, Ri, Ro])
        np.savez("/home/peter/Software/trd-standalone-tracking/Data/PbPb/MC/predictions/predictions_event%i"%i, X=X, Ri=Ri, Ro=Ro, y=y, preds=preds)
    model.save("/home/peter/Software/trd-standalone-tracking/Data/PbPb/MC/trained_model")
    model.save_weights("/home/peter/Software/trd-standalone-tracking/Data/PbPb/MC/trained_model/trained_model.h5")
    print(str(datetime.datetime.now()) + ': Training completed!')

