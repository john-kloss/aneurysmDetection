import tensorflow as tf
import numpy as np
import os
import progressbar
from datetime import datetime
import model

IMG_SIZE_PX = 32

n_classes = 2
batch_size = 10

saved_graph_path = os.getcwd()+'/data/tmp/log/'

keep_rate = 0.8


"""
This is the main function. It constructs the neural network, introduces a cost function
and trains an optimizer. 
Consecutively the network is trained with the train_data
@params: train_data, validation_data
"""


def train_neural_network(train_data, validation_data, name="train"):
    #tf.reset_default_graph()
    #saved_graph = tf.train.import_meta_graph(os.getcwd()+'/data/tmp/log/sess-36000.meta')


    with tf.name_scope(name):
        # with tf.device('/gpu:0'):

        prediction = model.convolutional_neural_network(x)

        #if os.path.isfile(saved_graph_path+"current_sess.meta") :
        #    tf.reset_default_graph()
        #new_saver = tf.train.import_meta_graph(saved_graph_path+"current_sess.meta")

            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
            )

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

        hm_epochs = 10
        _ew_saver = tf.train.Saver()
        with tf.Session() as sess:
            
            #new_saver.restore(sess,tf.train.latest_checkpoint(os.getcwd()+'/data/tmp/log/'))
           
            filename=os.getcwd()+"/data/tmp/log/"+datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
            writer = tf.summary.FileWriter(filename, sess.graph)
            sess.run(tf.global_variables_initializer())

            successful_runs = 0
            total_runs = 0

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for i in range(len(train_data["images"])):
                    total_runs += 1
                    try:
                        X = train_data["images"][i].reshape(1, 32, 32, 32)
                        Y = train_data["labels"][i].reshape(1, 2)
                        _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                        
                        #if (i*epoch) % 1000 == 0:
                         #   new_saver.save(sess, os.getcwd()+'/data/tmp/log/current_sess', global_step=i*epoch)

                        epoch_loss += c
                        successful_runs += 1
                    except Exception as e:
                        pass

                print(
                    "Epoch",
                    epoch + 1,
                    "completed out of",
                    hm_epochs,
                    "loss:",
                    epoch_loss,
                )

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, "float"))

                ev = []
                for i in range(len(validation_data["images"])):
                    # append the evaluation
                    ev.append(
                        accuracy.eval(
                            {
                                x: validation_data["images"][i].reshape(1, 32, 32, 32),
                                y: validation_data["labels"][i].reshape(1, 2),
                            }
                        )
                    )
                print(np.mean(ev))
                

            print("Done. Finishing accuracy:")

                print("fitment percent:", successful_runs / total_runs)
