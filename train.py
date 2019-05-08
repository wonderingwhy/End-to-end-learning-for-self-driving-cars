'''
You are going to train the CNN model here.

'''
import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import load_data
import model

model_path = './save'
logs_path = './log'
L2NormConst = 0.01

sess = tf.InteractiveSession()

train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.tf_Y, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

epochs = 50
batch_size = 128

# train
print("***************start training*******************")
for epoch in range(epochs):
    for i in range(int(load_data.num_train_images / batch_size)):
        xs, ys = load_data.LoadTrainBatch(batch_size)
        train_step.run(feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        
        if i % 10 == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint_path = os.path.join(model_path, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            loss_value = loss.eval(feed_dict = {model.tf_X: xs, model.tf_Y: ys})
            print("epoch: %d, step: %d, loss = %g" % (epoch, batch_size * i, loss_value))
        summary = merged_summary_op.eval(feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch * load_data.num_train_images / batch_size + i)
print("***************end training*******************")

# test
print("*******************start testing*********************")
sum_of_loss = 0
for i in range(int(load_data.num_val_images / batch_size)):
    xs, ys = load_data.LoadValBatch(batch_size)
    loss_value = loss.eval(feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 1.0})
    sum_of_loss += loss_value
    print("step = %d, sum_of_loss = %g" % (batch_size * i, sum_of_loss))
print("***************end testing*******************")

print("lambda = %g, average_of_loss = %g" % (L2NormConst, sum_of_loss / int(load_data.num_val_images / batch_size)))
    