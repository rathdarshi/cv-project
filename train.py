import tensorflow as tf


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  
  # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
  correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

  # Calculate distance from actual labels using cross entropy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
  # Take mean for total loss
  loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

  # The model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

  return logits, train_op, loss_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

  keep_prob_value = 0.75
  learning_rate_value = 0.001
  for epoch in range(epochs):
      # Create function to get batches
      total_loss = 0
      for X_batch, gt_batch in get_batches_fn(batch_size):

          loss, _ = sess.run([cross_entropy_loss, train_op],
          feed_dict={input_image: X_batch, correct_label: gt_batch,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})

          total_loss += loss;

      print("EPOCH {} ...".format(epoch + 1))
      print("Loss = {:.3f}".format(total_loss))
      print()


