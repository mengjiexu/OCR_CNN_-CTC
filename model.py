import tensorflow as tf

def build_model(num_classes):
    init_learning_rate = 0.001
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate,
                                               global_step,
                                               10000,
                                               0.9,
                                               staircase=True)
    inputs = tf.placeholder(tf.float32, [None, 40, None, 1])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    ################################CNN###################################
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    # Convolutional Layer #2 and Pooling Layer #2 19
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(pool2.get_shape())
    # Convolutional Layer #3 and Pooling Layer #3 9
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    print(pool3.get_shape())
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[2,1])
    print(pool4.get_shape())
    conv5 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 5],
      padding="valid",
      activation=tf.nn.relu)
    print(conv5.get_shape())
    ######################################################################
#     def lstm_cell():
#         return tf.contrib.rnn.LSTMCell(num_hidden)
#     stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, num_layers)],
#                                             state_is_tuple=True)

    # The second output is the last state and we will no use that
    features = tf.transpose(conv5, (0, 2, 1, 3))
    shape = tf.shape(features)
    features = tf.reshape(features, [shape[0], shape[1], 2*512])
    batch_s, max_timesteps = shape[0], shape[1]
#     outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(features, [-1, 1024])
    logits = tf.layers.dense(inputs=outputs, units = num_classes)
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    #######################################################################

    print(logits.get_shape())
    loss = tf.nn.ctc_loss(targets, logits, seq_len, preprocess_collapse_repeated=False, ctc_merge_repeated=False)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    # Accuracy: label error rate
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    return inputs, targets, seq_len, cost, optimizer, acc, decoded, global_step, learning_rate, log_prob