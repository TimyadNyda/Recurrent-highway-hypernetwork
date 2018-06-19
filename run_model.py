#simple example of running hyperrhncell (with tensorboard writer)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
from fs.osfs import OSFS
from rhn import HyperRHNCell

#flags to separate tensorboard files

flags = tf.app.flags
FLAGS = flags.FLAGS
logs_path = '/tmp/tensorflow_logs/example/'
flags.DEFINE_string('test_dir', logs_path, 'Directory to log testing.')

inputs_dim = 10
h = 100
m = 200
drop_out = 0.3
network_size = 5
batch_size = 1000

tf.reset_default_graph()
features_placeholder = tf.placeholder(tf.float32, shape=[None, None, 23])
labels_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
dataset = dataset.shuffle(len(b))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
iterator = dataset.make_initializable_iterator()
inputs, labels = iterator.get_next()


with tf.name_scope('Model'):
   
   rnn = hyperrhncell(h,m,network_size,inputs,drop_out, batch_norm = True)
   outputs = rnn.get_states()
   last_output = outputs_[-1]
   
   """to use whole sequence in last layer
   outputs = tf.transpose(outputs, perm=[1, 0, 2])
   outputs = tf.reshape(outputs,[-1,m*inputs.shape[1])
   """
   last_output = tf.layers.dense(last_output,50, tf.nn.relu)
   output = tf.layers.dense(last_output,2)

with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels,output))
    
with tf.name_scope('Optimizer'):

    lr = tf.placeholder(tf.float32, shape=[])
    
    optimizer = tf.train.AdamOptimizer(lr)#.minimize(loss)   
    optimizer = tf.train.AdamOptimizer(0.01)
    
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(,10)
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
    
with tf.name_scope('Accuracy'):
    
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))  
    accuracy = (tf.reduce_mean(tf.cast(correct, tf.float32))) *100


init = tf.global_variables_initializer()

# Write in summaries
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
    
# Merge all summaries 
merged_summary_op = tf.summary.merge_all()

training_epochs = 100       
            
folder = OSFS(FLAGS.test_dir)
test_n = len(list(n for n in folder.listdir('') if n.startswith('test')))
this_test = FLAGS.test_dir+"/test" + str(test_n+1)

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(this_test,
                                            graph=tf.get_default_graph())
    lr_ = 0.0001
    for tr_epoch in range(training_epochs):
                                        
        sess.run(iterator.initializer, feed_dict={features_placeholder: X_train,
                                          labels_placeholder: Labels})
        train_error = 0
        ac = 0
        iter = np.floor(len(X_train), batch_size)
        for epoch in range(iter): 
            _, cost, acc, summ  = sess.run([apply_grads, loss, accuracy, merged_summary_op], feed_dict={lr : lr_})
            ac += acc
            train_error += cost
            summary_writer.add_summary(summ, tr_epoch * 50 + epoch)
        train_error = train_error/(iter)
        ac = ac/(iter)
        print("Epoch:", '%02d' % (tr_epoch), "cost=", "{:.5f}".format(train_error))
        print("Accuracy:", "{:.5f}".format(ac))
