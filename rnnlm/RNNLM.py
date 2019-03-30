import tensorflow as tf
import numpy as np
import math

__all__ = [
    'RNNLM'
]


class RNNLM(object):
    def __init__(self,
                 vocab_size,
                 batch_size,
                 num_epochs,
                 check_point_step,
                 num_train_samples,
                 num_valid_samples,
                 num_layers,
                 num_hidden_units,
                 max_gradient_norm,
                 initial_learning_rate=1,
                 final_learning_rate=0.001):

        # Hyper Params Init
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.check_point_step = check_point_step
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.max_gradient_norm = max_gradient_norm

        # Util Variable and Ops Init.
        self.global_step = tf.Variable(0, trainable=False)

        # We set a dynamic learning rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.
        self.learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            self.global_step,
            decay_steps=150,
            decay_rate=0.96,
            staircase=True,
        )

        # Make LR no smaller than final_learning_rate.
        self.learning_rate = tf.cond(tf.less(self.learning_rate, final_learning_rate),
                                     lambda: tf.constant(final_learning_rate),
                                     lambda: self.learning_rate)

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        # Filenames for Datasets.
        self.filename_train = tf.placeholder(tf.string)
        self.filename_validation = tf.placeholder(tf.string)
        self.filename_test = tf.placeholder(tf.string)

        # Dataset Preparation.
        def parse(line):
            """
            Turn a line into an (input, output) pair.
            """
            # The sentence is <sos> 1 2 3 <eos>
            line_split = tf.string_split([line])
            # input is <sos> 1 2 3
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            # output is 1 2 3 <eos>
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            # The shapes of input_seq and output_seq are the same.
            # Both are [None].
            return input_seq, output_seq

        # The map(parse) actually create both input and output from the text lines.
        training_dataset = tf.data.TextLineDataset(self.filename_train).map(parse).shuffle(256).padded_batch(
            self.batch_size, padded_shapes=([None], [None]))

        # padded_batch() first pad each element and then batch them.
        # The padding is done according to padded_shapes, which should match the shape of the element.
        # Using None in dimension means padding to the max length of that dimension.

        # Don't shuffle validation dataset.
        validation_dataset = tf.data.TextLineDataset(self.filename_validation).map(parse).padded_batch(
            self.batch_size, padded_shapes=([None], [None]))

        # Note the batch(): each sentence become a batch.
        # Isn't it no batching at all?
        test_dataset = tf.data.TextLineDataset(self.filename_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)

        self.input_batch, self.output_batch = iterator.get_next()

        self.training_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)
        # End Dataset Preparation

        # Input Embedding Init.
        self.input_embedding_mat = tf.get_variable("input_embedding_mat",
                                                   [self.vocab_size, self.num_hidden_units],
                                                   dtype=tf.float32)

        self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch)

        # RNN Network Init
        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * self.num_layers, state_is_tuple=True)

        self.cell = cell

        # Output Embeddings Init
        self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                    [self.vocab_size, self.num_hidden_units],
                                                    dtype=tf.float32)

        self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                     [self.vocab_size],
                                                     dtype=tf.float32)

        # 0 => 0; non-zero => 1. shape is [B, M]
        non_zero_weights = tf.sign(self.input_batch)
        # all dimensions are reduced. valid_words is a scalar.
        # count the non-zero elements in [B, M].
        self.valid_words = tf.reduce_sum(non_zero_weights)

        # Compute sequence length
        def get_length(non_zero_place):
            # shape change: [B, M] => [B].
            # count the valid words in each sentence.
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        real_length_in_batch = get_length(non_zero_weights)

        # ================================================
        # Understanding How Predicting the Next Word Works
        # ================================================
        # Since the input and output are basically the same sequence, one maybe confused by
        # this setting, thinking that the RNN is cheating, knowing the answer before computing.
        #
        # No, the answer is in fact, RNN
        # is *predicting the next word* given the current word and its hidden state.
        # For each time step, w_i is feed as input, and the RNN combine its hidden state h_i and w_i
        # to output the next state h_{i+1} and w^{\hat}_{i+1}, which is an observation of the true next word
        # w_{i+1}. This observation, denoted as w_o for short, is compared against the ground truth next word
        # and SGD is applied to tune the parameters. This finish one prediction. On the next prediction, the
        # word previously being the ground truth now becomes input.
        # Whenever the RNN is doing a prediction, it really don't know the answer, although the answer will be
        # immediately feed as the next input.
        #
        # =====================
        # A Sample Illustration
        # =====================
        # We have a sentence "a b c d".
        # After padding it becomes: "<sos> a b c d <eos>".
        # A diagram showing what an RNN see as input and the corresponding output is:
        #
        # a b c d      ground truth
        #   b c d e    input
        # This is the trickiest part in understanding how RNNLM works, in my opinion.

        # The shape of outputs is [batch_size, max_length, num_hidden_units]
        # The shape of inputs is [B, M, H].
        # So the dynamic_rnn turns a sequence into another sequence.
        outputs, _ = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=self.input_embedded,
            sequence_length=real_length_in_batch,
            dtype=tf.float32
        )

        # Turn a sequence of vectors into a sequence of logits.
        # By mapping the vector of shape [H] to logits of shape [V] using
        # weight matrix of shape [H, V] and a bias vector of shape [V].
        def output_embedding(current_output):
            return tf.add(tf.matmul(
                current_output,
                tf.transpose(self.output_embedding_mat)
            ), self.output_embedding_bias)

        # To compute the logits
        # outputs shape: [B, M, H]
        # logits shape: [B, M, V]
        logits = tf.map_fn(output_embedding, outputs)

        # Flatten the logits from [B,M,V] to [B*M,V]
        logits = tf.reshape(logits, [-1, vocab_size])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            # Flatten the labels from [B,M] to [B*M]
            labels=tf.reshape(self.output_batch, [-1]),
            logits=logits,
        )
        # The shape of loss is [B*M].

        # zero out the loss of the padding words.
        self.loss = loss * tf.cast(
            # Flatten non_zero_weights from [B, M] to [B*M] to match the shape of loss.
            tf.reshape(non_zero_weights, [-1]),
            tf.float32,
        )

        # Train
        params = tf.trainable_variables()

        opt = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # This is the Training Ops.
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def batch_train(self, sess, saver):
        best_score = np.inf
        patience = 5
        epoch = 0

        while epoch < self.num_epochs:
            # Init the training iterator.
            sess.run(self.training_init_op, {self.filename_train: "./data/train.ids"})
            train_loss = 0.0
            train_valid_words = 0

            # Iterate over the dataset.
            while True:
                try:
                    _loss, _valid_words, global_step, current_learning_rate, _ = sess.run(
                        [self.loss, self.valid_words, self.global_step, self.learning_rate, self.updates],
                        {self.dropout_rate: 0.5})
                    train_loss += np.sum(_loss)
                    train_valid_words += _valid_words

                    if global_step % self.check_point_step == 0:
                        train_loss /= train_valid_words
                        train_ppl = math.exp(train_loss)
                        print("Training Step: {}, LR: {}".format(global_step, current_learning_rate))
                        print("    Training PPL: {}".format(train_ppl))
                        train_loss = 0.0
                        train_valid_words = 0
                except tf.errors.OutOfRangeError:
                    break  # The end of one epoch

            # Run validation after one epoch.
            sess.run(self.validation_init_op, {self.filename_validation: "./data/valid.ids"})
            dev_loss = 0.0
            dev_valid_words = 0
            while True:
                try:
                    _dev_loss, _dev_valid_words = sess.run(
                        [self.loss, self.valid_words],
                        # Remember: no dropout in testing or validation.
                        {self.dropout_rate: 1.0})

                    dev_loss += np.sum(_dev_loss)
                    dev_valid_words += _dev_valid_words
                except tf.errors.OutOfRangeError:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print("Validation PPL: {}".format(dev_ppl))

                    # If the dev_ppl don't get any better after 5 epoch, the training
                    # ends in advance.
                    if dev_ppl < best_score:
                        patience = 5
                        saver.save(sess, "model/best_model.ckpt")
                        best_score = dev_ppl
                    else:
                        patience -= 1
                    if patience == 0:
                        epoch = self.num_epochs
                    break

    def predict(self, sess, input_file, raw_file, verbose=False):
        """

        :param sess:
        :param input_file: gap filling exercise ids file.
        :param raw_file: gap filling exercise file.
        :param verbose:
        :return:
        """
        # if verbose is true, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.filename_test: input_file})

        with open(raw_file) as fp:
            global_dev_loss = 0.0
            global_dev_valid_words = 0

            for raw_line in fp.readlines():
                raw_line = raw_line.strip()
                # Note the update ops is still built, but not run!
                _dev_loss, _dev_valid_words, input_line = sess.run(
                    [self.loss, self.valid_words, self.input_batch],
                    {self.dropout_rate: 1.0}
                )

                dev_loss = np.sum(_dev_loss)
                dev_valid_words = _dev_valid_words

                global_dev_loss += dev_loss
                global_dev_valid_words += dev_valid_words

                if verbose:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print(raw_line + "    Test PPL: {}".format(dev_ppl))

            global_dev_loss /= global_dev_valid_words
            global_dev_ppl = math.exp(global_dev_loss)
            print("Global Test PPL: {}".format(global_dev_ppl))
