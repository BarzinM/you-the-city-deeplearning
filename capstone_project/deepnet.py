import tensorflow as tf
import numpy as np

report_step = 100
decay = .95
initial_learning_rate = 10.

class MnistTrainer(object):
    def __init__(self):
        pass

    def makeGraph(self):
        image_size = 28
        num_channels = 1
        num_classes = 5
        graph = tf.Graph()
        with graph.as_default():
            # placeholders for data
            data_flow = tf.placeholder(tf.float32, shape=(
                batch_size, image_size, 5*image_size, num_channels),name="data_flow_placeholder")
            label_flow = tf.placeholder(
                tf.float32, shape=(batch_size, num_classes),name="label_flow_placeholder")

            # dropout ratio
            keep_prob = tf.placeholder(tf.float32)

            # variables
            # used for decaying the leraning rate
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate, global_step, report_step, decay)

            weight_1 = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
            bias_1 = tf.Variable(tf.constant(.05, shape=[depth_1]))

            weight_2 = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
            bias_2 = tf.Variable(tf.constant(.05, shape=[depth_2]))

            new_image_size = image_size // 4
            weight_3 = tf.Variable(tf.truncated_normal(
                [5*new_image_size**2 * depth_2, number_of_hidden], stddev=0.1))
            bias_3 = tf.Variable(tf.constant(.05, shape=[number_of_hidden]))

            weight_4 = tf.Variable(tf.truncated_normal(
                [number_of_hidden, num_classes], stddev=0.1))
            bias_4 = tf.Variable(tf.constant(.05, shape=[num_classes]))

            # constants for test dataset
            # variables for convolutional,bias,

            def generateLogit(data):
                # first convolutional
                data = tf.nn.conv2d(data, weight_1, [1, 1, 1, 1], padding='SAME')
                # max pool
                data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                      1, 2, 2, 1], padding='SAME')
                # relu
                data = tf.nn.relu(data + bias_1)
                # droped
                data = tf.nn.dropout(data, keep_prob)

                # second convolutional
                data = tf.nn.conv2d(data, weight_2, [1, 1, 1, 1], padding='SAME')
                # max pool
                data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                      1, 2, 2, 1], padding='SAME')
                # relu
                data = tf.nn.relu(data + bias_2)
                # droped
                data = tf.nn.dropout(data, keep_prob)

                # reshape to 2D [batch size, all features]
                shape = data.get_shape().as_list()
                data = tf.reshape(data, [-1, shape[1] * shape[2] * shape[3]])
                # relu
                data = tf.nn.relu(tf.matmul(data, weight_3) + bias_3)

                # droped
                data = tf.nn.dropout(data, keep_prob)
                # relu
                data = tf.nn.relu(tf.matmul(data, weight_4) + bias_4)

                # # droped
                # data = tf.nn.dropout(data, keep_prob)
                # # logits
                # data = tf.nn.relu(tf.matmul(data, weight_5) + bias_5)

                return data

            logits = generateLogit(data_flow)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, label_flow))
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(loss, global_step=global_step)

            predicts = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(predicts, 1), tf.argmax(label_flow, 1))
            performance = tf.reduce_sum(tf.cast(correct, "float"))

    def train(self):
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()

            for step in range(num_steps):
                train_data, train_labels = generator(batch_size)
                feed_dict = {data_flow: train_data,
                             label_flow: train_labels,
                             keep_prob: .7}
                _, loss_return = session.run(
                    [optimizer, loss], feed_dict=feed_dict)

                if step % report_step == 0:
                    print("loss",loss_return)
                    valid_accuracy = 0
                    for batch in range(0, validation_batches, batch_size):
                        valid_data, valid_labels = generator(batch_size)
                        valid_accuracy += batchCorrects(valid_data, valid_labels,model_pack)
                    valid_accuracy = 100*valid_accuracy / validation_batches
                    print('Validation accuracy: %.1f%%' % valid_accuracy)

            test_accuracy = 0
            for batch in range(0, test_batches, batch_size):
                valid_data, valid_labels = generator(batch_size)
                test_accuracy += batchCorrects(valid_data, valid_labels,model_pack)
            test_accuracy = 100*test_accuracy / test_batches
            print('Test accuracy: %.1f%%' % test_accuracy)

    def batchCorrects(self):
        length = labels.shape[0]
        correctness = 0
        for batch_start in range(0, length, batch_size):
            feed_dict = {data_flow: data[batch_start:(batch_start + batch_size)],
                         label_flow: labels[batch_start:(batch_start + batch_size)],
                         keep_prob: 1.0}  # batch data and batch labels and keep_prob
            correctness += performance.eval(feed_dict=feed_dict)

        return correctness

def generateGraph(batch_size, patch_size, depth_1, depth_2, number_of_hidden, number_of_hidden_2):
    image_size = 28
    num_channels = 1
    num_classes = 5
    graph = tf.Graph()
    with graph.as_default():
        # placeholders for data
        data_flow = tf.placeholder(tf.float32, shape=(
            batch_size, image_size, 5*image_size, num_channels),name="data_flow_placeholder")
        label_flow = tf.placeholder(
            tf.float32, shape=(batch_size, num_classes),name="label_flow_placeholder")

        # dropout ratio
        keep_prob = tf.placeholder(tf.float32)

        # variables
        # used for decaying the leraning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, report_step, decay)

        weight_1 = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth_1]))
        bias_1 = tf.Variable(tf.constant(.05, shape=[depth_1]))

        weight_2 = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth_1, depth_2]))
        bias_2 = tf.Variable(tf.constant(.05, shape=[depth_2]))

        new_image_size = image_size // 4
        weight_3 = tf.Variable(tf.truncated_normal(
            [5*new_image_size**2 * depth_2, number_of_hidden]))
        bias_3 = tf.Variable(tf.constant(.05, shape=[number_of_hidden]))

        weight_4 = tf.Variable(tf.truncated_normal(
            [number_of_hidden, num_classes]))
        bias_4 = tf.Variable(tf.constant(.05, shape=[num_classes]))

        # constants for test dataset
        # variables for convolutional,bias,

        def generateLogit(data):
            # first convolutional
            data = tf.nn.conv2d(data, weight_1, [1, 1, 1, 1], padding='SAME')
            # max pool
            data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                  1, 2, 2, 1], padding='SAME')
            # relu
            data = tf.nn.relu(data + bias_1)
            # droped
            data = tf.nn.dropout(data, keep_prob)

            # second convolutional
            data = tf.nn.conv2d(data, weight_2, [1, 1, 1, 1], padding='SAME')
            # max pool
            data = tf.nn.max_pool(data, [1, 2, 2, 1], [
                                  1, 2, 2, 1], padding='SAME')
            # relu
            data = tf.nn.relu(data + bias_2)
            # droped
            data = tf.nn.dropout(data, keep_prob)

            # reshape to 2D [batch size, all features]
            shape = data.get_shape().as_list()
            data = tf.reshape(data, [-1, shape[1] * shape[2] * shape[3]])
            # relu
            data = tf.nn.relu(tf.matmul(data, weight_3) + bias_3)

            # droped
            data = tf.nn.dropout(data, keep_prob)
            # relu
            data = tf.nn.relu(tf.matmul(data, weight_4) + bias_4)

            # # droped
            # data = tf.nn.dropout(data, keep_prob)
            # # logits
            # data = tf.nn.relu(tf.matmul(data, weight_5) + bias_5)

            return data

        logits = generateLogit(data_flow)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, label_flow))
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)

        predicts = tf.nn.softmax(logits)
        correct = tf.equal(tf.argmax(predicts, 1), tf.argmax(label_flow, 1))
        performance = tf.reduce_sum(tf.cast(correct, "float"))

        return (graph,batch_size,data_flow,label_flow,keep_prob,optimizer,loss,performance)


def batchCorrects(data, labels, model_pack):
    graph,batch_size,data_flow,label_flow,keep_prob,optimizer,loss,performance=model_pack
    length = labels.shape[0]
    correctness = 0
    for batch_start in range(0, length, batch_size):
        feed_dict = {data_flow: data[batch_start:(batch_start + batch_size)],
                     label_flow: labels[batch_start:(batch_start + batch_size)],
                     keep_prob: 1.0}  # batch data and batch labels and keep_prob
        correctness += performance.eval(feed_dict=feed_dict)

    return correctness


def batchedAccuracy(data, labels, model_pack):
    correctness = batchCorrects(data, labels, model_pack)
    accuracy = correctness / labels.shape[0]
    return accuracy * 100


# def randomWithGap(high, size, gap_start, gap_end):
#     random_index = np.random.randint(high, size=size)
#     gap_size = gap_end - gap_start


def kFold(fold, data_file, labels_file, total_folds):
    labels = np.load(labels_file)
    data_size = labels.shape[0]
    gap_size = data_size / total_folds
    valid_labels = labels[(fold * gap_size):(fold + 1) * gap_size]
    labels = np.concatenate(
        [labels[0:fold * gap_size], labels[(fold + 1) * gap_size:]])
    data = np.load(data_file)
    valid_data = data[(fold * gap_size):(fold + 1) * gap_size]
    data = np.concatenate(
        [data[0:fold * gap_size], data[(fold + 1) * gap_size:]])
    return data, labels, valid_data, valid_labels


def shuffleAndSave(arrays, files):
    ind = np.arange(labels.shape[0])
    np.random.shuffle(ind)
    for i in range(len(files)):
        np.save(files[i], arrays[i][ind])
    return data, labels


def foldTrain(num_steps):
    number_of_folds = 10
    fold_size = number_of_data // 10
    iteration_in_each_fold = num_steps // number_of_folds
    data_set = shuffle(data_set)
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for fold in range(number_of_folds):
            train_data, train_labels, valid_data, valid_labels = kFold(
                fold, 'data_file', 'label_file', number_of_folds)
            offset_mod = train_labels.shape[0] - batch_size
            for step in range(iteration_in_each_fold):
                batch_start = (batch_size * step) % offset_mod
                batch_end = batch_size * (step + 1)
                feed_dict = {data_flow: train_data[batch_start:batch_end],
                             label_flow: train_labels[batch_start:batch_end],
                             keep_prob: .7}
                _, loss_return = session.run(
                    [optimizer, loss], feed_dict=feed_dict)
                if step % report_step == 0:
                    print('Validation accuracy: %.1f%%' %
                          batchAccuracy(valid_dataset, valid_labels))
        print('Test accuracy: %.1f%%' %
              batchAccuracy(test_dataset, test_labels))


def onlineTrain(model_pack, num_steps, generator, validation_batches, test_batches):
    graph,batch_size,data_flow,label_flow,keep_prob,optimizer,loss,performance=model_pack
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        for step in range(num_steps):
            train_data, train_labels = generator(batch_size)
            feed_dict = {data_flow: train_data,
                         label_flow: train_labels,
                         keep_prob: .7}
            _, loss_return = session.run(
                [optimizer, loss], feed_dict=feed_dict)

            if step % report_step == 0:
                print("loss",loss_return)
                valid_accuracy = 0
                for batch in range(0, validation_batches, batch_size):
                    valid_data, valid_labels = generator(batch_size)
                    valid_accuracy += batchCorrects(valid_data, valid_labels,model_pack)
                valid_accuracy = 100*valid_accuracy / validation_batches
                print('Validation accuracy: %.1f%%' % valid_accuracy)

        test_accuracy = 0
        for batch in range(0, test_batches, batch_size):
            valid_data, valid_labels = generator(batch_size)
            test_accuracy += batchCorrects(valid_data, valid_labels,model_pack)
        test_accuracy = 100*test_accuracy / test_batches
        print('Test accuracy: %.1f%%' % test_accuracy)
