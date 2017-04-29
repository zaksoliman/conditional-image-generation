import os
import scipy.misc
import numpy as np
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "../../../datasets/coco/train2014", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("z_dist", "gaussian", "Distribution to sample noise from [gaussian]")
flags.DEFINE_float('lr', 0.01, "Learning rate for optimizing the inpainting loss [0.01]")
flags.DEFINE_float('momentum',0.9, "Momentum for optimization of inpainting loss [0.9]")
flags.DEFINE_integer('n_iter', 1000)
flags.DEFINE_float('l_param', 0.1,"Weighting parameter for inpainting loss [0.1]")
flags.DEFINE_string('out_dir', 'inpainting')
FLAGS = flags.FLAGS

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, z_dist=FLAGS.z_dist, checkpoint_dir=FLAGS.checkpoint_dir,
            l_param=FLAGS.l_param)
    dcgan.complete(FLAGS)
