import os
import logging
import tensorflow as tf

from rnnlm.data_prepare import gen_vocab
from rnnlm.data_prepare import gen_id_seqs
from rnnlm.RNNLM import RNNLM

# It seems that there are some little bugs in tensorflow 1.4.1.
# You can find more details in
# https://github.com/tensorflow/tensorflow/issues/12414
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set TRAIN to true will build a new model
TRAIN = True

# If VERBOSE is true, then print the ppl of every sequence when we
# are testing.
VERBOSE = True

# To indicate your test corpus
test_file = "./gap_filling_exercise/gap_filling_exercise"
logging.info('test_file: %s', test_file)

if not os.path.isfile("data/vocab"):
    logging.info('generating vocab file...')
    gen_vocab("ptb/train")

if not os.path.isfile("data/train.ids"):
    logging.info('generating ids files...')
    gen_id_seqs("ptb/train")
    gen_id_seqs("ptb/valid")

logging.info('counting statistics of the dataset...')
with open("data/train.ids") as fp:
    num_train_samples = len(fp.readlines())
logging.info('num_train_samples: %d', num_train_samples)

with open("data/valid.ids") as fp:
    num_valid_samples = len(fp.readlines())
logging.info('num_valid_samples: %d', num_valid_samples)

with open("data/vocab") as vocab:
    vocab_size = len(vocab.readlines())
logging.info('vocab_size: %d', vocab_size)


def create_model(sess):
    model = RNNLM(
        vocab_size=vocab_size,
        batch_size=64,
        num_epochs=80,
        check_point_step=100,
        num_train_samples=num_train_samples,
        num_valid_samples=num_valid_samples,
        num_layers=2,
        num_hidden_units=600,
        initial_learning_rate=1.0,
        final_learning_rate=0.0005,
        max_gradient_norm=5.0,
    )

    logging.info('created model %r', model)
    sess.run(tf.global_variables_initializer())
    return model


if TRAIN:
    # Train the model.
    logging.info('training begins...')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(sess)
        saver = tf.train.Saver()
        model.batch_train(sess, saver)

# Open a fresh default graph for prediction.
tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)

# Test.
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = create_model(sess)
    saver = tf.train.Saver()
    logging.info('restoring model...')
    saver.restore(sess, "model/best_model.ckpt")

    predict_id_file = os.path.join("data", test_file.split("/")[-1] + ".ids")
    logging.info('using test_file: %s', predict_id_file)

    if not os.path.isfile(predict_id_file):
        logging.info('generating ids file...')
        gen_id_seqs(test_file)

    logging.info('prediction begins')
    model.predict(sess, predict_id_file, test_file, verbose=VERBOSE)
