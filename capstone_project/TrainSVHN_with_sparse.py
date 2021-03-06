import tensorflow as tf
import numpy as np
from time import strftime, time, localtime


def product(num_list):
    prod = 1
    for number in num_list:
        prod *= number
    return prod


class SVHNTrainer(object):
    def __init__(self):
        self.report_step = None
        self.decay = .95
        self.initial_learning_rate = None
        self.batch_size = None
        self.num_channels = None
        self.output_neurons = None
        self.image_height = None
        self.image_width = None
        self.depth_conv = None
        self.depth_fully_connected = None
        self.max_pool_strides = None
        self.label_sizes = [5]

        self.report_string = '-' * 10
        self.graph = tf.Graph()

    def report(self, *args):
        text = [str(arg) for arg in args]
        text = " ".join(text)
        self.report_string += "\n" + text
        print(text)

    def saveReport(self):
        file_name = "model_report_" + strftime("%Y-%m-%d_%H:%M:%S")

        text = "=" * 10
        text += "\n" + file_name
        text += "\nInit learing rate: " + str(self.initial_learning_rate)
        text += "\nDecay ratio: " + str(self.decay)
        text += "\nDecay step: " + str(self.report_step)
        text += "\nBatch size: " + str(self.batch_size)
        text += "\nDepth conv: " + str(self.depth_conv)
        text += "\nDepth fully connected: " + str(self.depth_fully_connected)

        text += "\n"
        text += self.report_string

        with open(file_name, "w+") as file_handle:
            file_handle.write(text)
        print("Saved to:", file_name)

    def makeGraph(self):

        print(vars(self))
        patch_size = 5

        def makeWeightAndBias(shape):
            weight = tf.Variable(tf.truncated_normal(shape, stddev=.01))
            bias = tf.Variable(tf.constant(.01, shape=[shape[-1]]))
            return weight, bias

        with self.graph.as_default():

            # placeholders for data
            self.data_flow = tf.placeholder(tf.float32, shape=(
                self.batch_size, self.image_height, self.image_width, self.num_channels), name="data_flow_placeholder")
            self.label_flow2 = [tf.placeholder(tf.float, shape=(
                self.batch_size, 1)) for _ in range(len(self.label_sizes))]
            # self.label_flow = tf.placeholder(
            # tf.float32, shape=(self.batch_size, self.output_neurons),
            # name="label_flow_placeholder")

            # dropout ratio
            self.keep_prob = tf.placeholder(tf.float32)

            # learning variables
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, global_step, self.report_step, self.decay)

            # defining convolutional weights and biases
            weights_conv = []
            biases_conv = []
            number_of_conv_layers = len(self.depth_conv)
            self.depth_conv = [self.num_channels] + self.depth_conv
            print("Weights of convolution layers have the following sizes:")
            for i in range(number_of_conv_layers):
                weight, bias = makeWeightAndBias(
                    [patch_size, patch_size, self.depth_conv[i], self.depth_conv[i + 1]])
                print(weight.get_shape(), "maxpool stride:",
                      self.max_pool_strides[i])
                weights_conv.append(weight)
                biases_conv.append(bias)

            # defining fully connected weights and biases
            new_image_height = self.image_height // product(
                self.max_pool_strides)
            new_image_width = self.image_width // product(
                self.max_pool_strides)
            weights_fully_connected = []
            biases_fully_connected = []
            number_of_fully_connected_layers = len(self.depth_fully_connected)
            self.depth_fully_connected = [
                new_image_height * new_image_width * self.depth_conv[-1]] + self.depth_fully_connected
            for i in range(number_of_fully_connected_layers):
                weight, bias = makeWeightAndBias(
                    [self.depth_fully_connected[i], self.depth_fully_connected[i + 1]])
                weights_fully_connected.append(weight)
                biases_fully_connected.append(bias)

            # configuring the output layer
            last_weights = []
            last_biases = []
            for label_size in self.label_sizes:
                weight, bias = makeWeightAndBias(
                    [self.depth_fully_connected[-1], label_size])
                last_weights.append(weight)
                last_biases.appnd(bias)
            # last_weight, last_bias = makeWeightAndBias(
            #     [self.depth_fully_connected[-1], self.output_neurons])

            def generateLogit(data):
                # convolutional layer operations
                print('Data has the following shapes between convolutional layers:')
                for weight, bias, stride in zip(weights_conv, biases_conv, self.max_pool_strides):
                    print(data.get_shape())
                    data = tf.nn.conv2d(
                        data, weight, [1, 1, 1, 1], padding='SAME')
                    data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                          1, stride, stride, 1], padding='SAME')
                    # data = tf.nn.dropout(data, self.keep_prob)
                    data = tf.nn.relu(data + bias)

                # rehape operation to connect convolutional to flat layers
                shape = data.get_shape().as_list()
                data = tf.reshape(
                    data, [self.batch_size, shape[1] * shape[2] * shape[3]])
                print('Data\'s shape before flat layers is:', data.get_shape())

                # fully connected layers operations
                print('Data has following shapes between flat layers:')
                for weight, bias in zip(weights_fully_connected, biases_fully_connected):
                    data = tf.nn.relu(tf.matmul(data, weight) + bias)
                    data = tf.nn.dropout(data, self.keep_prob)
                    print(data.get_shape())

                # output operations
                output = []
                print("Final shape of data is:", end=' ')
                for weight, bias in zip(last_weights, last_biases):
                    label = tf.nn.relu(tf.matmul(data, weight) + bias)
                    output.append(label)
                    print(label.get_shape(), end=' ')
                print()

                return output

            self.logits = generateLogit(self.data_flow)

            # def splitColumns(tensor, sizes):
            #     start = 0
            #     sliced_tensors = []
            #     for size in sizes:
            #         sliced_tensors.append(
            #             tf.slice(tensor, [0, start], [self.batch_size, size]))
            #         start += size
            #     return sliced_tensors

            # seperated_logits = splitColumns(self.logits, self.class_sizes)
            # seperated_labels = splitColumns(self.label_flow, self.class_sizes)
            # print("Seperated shapes are", [
            #       label.get_shape().as_list() for label in seperated_labels])

            # i = tf.constant(0, tf.int64)
            # condition = lambda i,l: tf.less(i,l)
            # variables = [i,tf.argmax(seperated_labels[0],1)]
            # loss = tf.constant(0.0)
            # def operation(i,l):
            #     logit = seperated_logits[i]
            #     label = seperated_labels[i]
            #     loss+=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit,label))
            # tf.while_loop(condition,operation,variables)
            # self.loss = loss

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits[0], self.label_flow2[0])

            # def multiLabelLoss(logits_list, labels_list):
            #     loss = tf.constant(0.0)
            #     for i in range(len(logits_list)):
            #         logit = logits_list[i]
            #         label = labels_list[i]
            #         loss += tf.reduce_mean(
            #             tf.nns.sparse_softmax_cross_entropy_with_logits(logit, label))
            #     return loss

            # self.loss = multiLabelLoss(seperated_logits, seperated_labels)
            # print("loss shape is",loss.get_shape())

            # self.loss = tf.reduce_mean(
            # tf.nn.softmax_cross_entropy_with_logits(logits, self.label_flow))

            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(self.loss, global_step=global_step)

            performance = []
            for label in range(len(self.label_sizes[0])):
                predicts = tf.nn.softmax(seperated_logits[label])
                correct = tf.equal(tf.argmax(predicts, 1),
                                   tf.argmax(seperated_labels[label], 1))
                performance.append(tf.reduce_sum(
                    tf.cast(correct, "float"), keep_dims=True))
            self.performance = tf.concat(0, performance)

            # self.performance = 0
            # for logit,label in zip(seperated_logits,seperated_labels):
            #     predicts = tf.nn.softmax(logit)
            #     correct = tf.equal(tf.argmax(predicts, 1),
            #                        tf.argmax(label, 1))
            #     self.performance += tf.reduce_sum(tf.cast(correct, "float"))

    def train(self, training_steps, generator, validation_steps, test_steps):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()

            start_time = time()
            for step in range(1, training_steps + 1):

                train_data, train_labels = next(generator)

                feed_dict = {i: d for i, d in zip(
                    self.label_flow2, train_labels)}
                feed_dict[self.data_flow] = train_data
                feed_dict[self.keep_prob] = .7
                # feed_dict = {self.data_flow: train_data,
                #              self.label_flow: train_labels,
                #              self.keep_prob: .7}
                _, loss_return = session.run(
                    [self.optimizer, self.loss], feed_dict=feed_dict)

                if step % self.report_step == 0:
                    current_time = time()
                    self.report("self.loss", loss_return)
                    detailed_eval = np.zeros((len(self.label_sizes)))
                    for batch in range(0, validation_steps):
                        valid_data, valid_labels = next(generator)
                        feed_dict = {self.data_flow: valid_data,
                                     self.label_flow: valid_labels,
                                     self.keep_prob: 1.0}  # batch data and batch labels and self.keep_prob
                        detailed_eval += self.performance.eval(feed_dict)

                    detailed_eval = 100 * detailed_eval / \
                        (self.batch_size * validation_steps)
                    self.report('Validation accuracy', detailed_eval)
                    elapsed_time = current_time - start_time
                    print("Next report at:", strftime(
                        "%H:%M:%S", localtime(current_time + elapsed_time)))

            detailed_eval = np.zeros((len(self.label_sizes)))
            for batch in range(0, test_steps):
                test_data, test_labels = next(generator)
                feed_dict = {self.data_flow: test_data,
                             self.label_flow: test_labels,
                             self.keep_prob: 1.0}  # batch data and batch labels and self.keep_prob
                detailed_eval += self.performance.eval(feed_dict)

            detailed_eval = 100 * detailed_eval / \
                (self.batch_size * test_steps)
            self.report('Test accuracy:', detailed_eval)

            feed_dict = {self.data_flow: test_data,
                         self.label_flow: test_labels,
                         self.keep_prob: 1.0}  # batch data and batch labels and self.keep_prob

            return test_data, self.logits.eval(feed_dict=feed_dict)

    def batchEvaluation(self, data, labels):
        length = labels.shape[0]
        correctness = 0
        for batch_start in range(0, length, self.batch_size):
            feed_dict = {self.data_flow: data[batch_start:(batch_start + self.batch_size)],
                         self.label_flow: labels[batch_start:(batch_start + self.batch_size)],
                         self.keep_prob: 1.0}  # batch data and batch labels and self.keep_prob
            # correctness += self.performance.eval(feed_dict=feed_dict)
            for performance in self.performance:
                correctness += performance.eval(feed_dict)

        return correctness / len(self.label_sizes)
