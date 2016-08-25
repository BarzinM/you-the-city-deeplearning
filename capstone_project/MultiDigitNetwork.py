import tensorflow as tf
import numpy as np


class MnistTrainer(object):
    def __init__(self):
        self.report_step = 500
        self.decay = .95
        self.initial_learning_rate = 1e-4

    def makeGraph(self):
        self.batch_size = 8
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 5
        # number_of_hidden = 5
        patch_size = 8
        depth_1 = 40
        # depth_2 = 11
        self.graph = tf.Graph()
        with self.graph.as_default():
            # placeholders for data
            self.data_flow = tf.placeholder(tf.float32, shape=(
                self.batch_size, self.image_size, 5 * self.image_size, self.num_channels), name="data_flow_placeholder")
            self.label_flow = tf.placeholder(
                tf.float32, shape=(self.batch_size, self.num_classes), name="label_flow_placeholder")

            # dropout ratio
            self.keep_prob = tf.placeholder(tf.float32)

            # variables
            # used for decaying the leraning rate
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, global_step, self.report_step, self.decay)

            weight_1 = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, self.num_channels, depth_1], stddev=0.1))
            bias_1 = tf.Variable(tf.constant(.05, shape=[depth_1]))

            # weight_2 = tf.Variable(tf.truncated_normal(
            #     [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
            # bias_2 = tf.Variable(tf.constant(.05, shape=[depth_2]))

            new_image_size = self.image_size // 2
            weight_3 = tf.Variable(tf.truncated_normal(
                [5 * new_image_size**2 * depth_1, self.num_classes], stddev=0.1))
            bias_3 = tf.Variable(tf.constant(.05, shape=[self.num_classes]))

            # weight_4 = tf.Variable(tf.truncated_normal(
            #     [number_of_hidden, self.num_classes], stddev=0.1))
            # bias_4 = tf.Variable(tf.constant(.05, shape=[self.num_classes]))

            # constants for test dataset
            # variables for convolutional,bias,

            def generateLogit(data):
                # first convolutional
                data = tf.nn.conv2d(
                    data, weight_1, [1, 1, 1, 1], padding='SAME')
                # max pool
                data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                      1, 2, 2, 1], padding='SAME')
                # relu
                data = tf.nn.relu(data + bias_1)
                # droped
                # data = tf.nn.dropout(data, self.keep_prob)

                # second convolutional
                # data = tf.nn.conv2d(
                #     data, weight_2, [1, 1, 1, 1], padding='SAME')
                # max pool
                # data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                      # 1, 2, 2, 1], padding='SAME')
                # relu
                # data = tf.nn.relu(data + bias_2)
                # droped
                # data = tf.nn.dropout(data, self.keep_prob)

                # reshape to 2D [batch size, all features]
                shape = data.get_shape().as_list()
                data = tf.reshape(data, [self.batch_size, shape[1] * shape[2] * shape[3]])
                # relu
                data = tf.nn.relu(tf.matmul(data, weight_3) + bias_3)

                # droped
                # data = tf.nn.dropout(data, self.keep_prob)
                # relu
                # data = tf.nn.relu(tf.matmul(data, weight_4) + bias_4)

                # # droped
                # data = tf.nn.dropout(data, keep_prob)
                # # logits
                # data = tf.nn.relu(tf.matmul(data, weight_5) + bias_5)

                print(data.get_shape().as_list())
                return data

            logits = generateLogit(self.data_flow)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.label_flow))
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(self.loss, global_step=global_step)

            predicts = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(predicts, 1),
                               tf.argmax(self.label_flow, 1))
            self.performance = tf.reduce_sum(tf.cast(correct, "float"))

    def train(self, num_steps, generator, validation_batches, test_batches):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()

            for step in range(num_steps):
                train_data, train_labels = generator(self.batch_size)
                feed_dict = {self.data_flow: train_data,
                             self.label_flow: train_labels,
                             self.keep_prob: 1.}
                _, loss_return = session.run(
                    [self.optimizer, self.loss], feed_dict=feed_dict)

                if step % self.report_step == 0:
                    print("self.loss", loss_return)
                    valid_accuracy = 0
                    for batch in range(0, validation_batches, self.batch_size):
                        valid_data, valid_labels = generator(self.batch_size)
                        valid_accuracy += self.batchCorrects(
                            valid_data, valid_labels)
                    valid_accuracy = 100 * valid_accuracy / validation_batches
                    print('Validation accuracy: %.1f%%' % valid_accuracy)

            test_accuracy = 0
            for batch in range(0, test_batches, self.batch_size):
                valid_data, valid_labels = generator(self.batch_size)
                test_accuracy += self.batchCorrects(valid_data, valid_labels)
            test_accuracy = 100 * test_accuracy / test_batches
            print('Test accuracy: %.1f%%' % test_accuracy)

    def batchCorrects(self, data, labels):
        length = labels.shape[0]
        correctness = 0
        for batch_start in range(0, length, self.batch_size):
            feed_dict = {self.data_flow: data[batch_start:(batch_start + self.batch_size)],
                         self.label_flow: labels[batch_start:(batch_start + self.batch_size)],
                         self.keep_prob: 1.0}  # batch data and batch labels and self.keep_prob
            correctness += self.performance.eval(feed_dict=feed_dict)

        return correctness
