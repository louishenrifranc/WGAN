from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, conv2d, conv2d_transpose, fully_connected
from argparse import ArgumentParser
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import utils

tf.flags.DEFINE_integer("batch_size", 32, "Size of a batch")
tf.flags.DEFINE_integer("hidden_dim", 100, "Latent space dimension")
tf.flags.DEFINE_integer("input_size", 28, "Dimension of input")
tf.flags.DEFINE_integer("nb_iter", 10000, "Number of iterations")
tf.flags.DEFINE_integer("critic_iter", 50, "Number of critic iterations")
tf.flags.DEFINE_string("model_folder", "model", "Where the model is saved")

tf.flags.DEFINE_integer("nb_classes", 10, "Number of different target classes")
FLAGS = tf.flags.FLAGS

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=False)


class GAN:
    def __init__(self, flags=FLAGS):
        self.batch_size = flags.batch_size
        self.input_size = flags.input_size ** 2
        self.latent_dim = flags.hidden_dim
        self.nb_targets = flags.nb_classes
        self.optimizer = tf.train.RMSPropOptimizer(2e-4)

        self.nb_iter = flags.nb_iter
        self.nb_critic_iter = flags.critic_iter

        self.sess = tf.Session()

    def _create_placeholder(self, scope_name="placeholder"):
        with tf.variable_scope(scope_name):
            self.x = tf.placeholder(tf.float32, [None, self.input_size])
            self.z = tf.placeholder(tf.float32, [None, self.latent_dim])
            self.y = tf.placeholder(tf.float32, [None, self.nb_targets])
            self.is_training = tf.placeholder(tf.bool)

    def _feed_dict(self, is_training, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        batch_iterator = mnist.train if is_training else mnist.test
        batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, self.latent_dim]).astype(np.float32)
        batch_x, batch_y = batch_iterator.next_batch(batch_size)
        index = np.random.permutation(batch_x.shape[0])
        batch_x, batch_y = batch_x[index], batch_y[index]
        batch_x = 2 * batch_x - 1
        return {self.z: batch_z,
                self.x: batch_x,
                self.y: batch_y,
                self.is_training: is_training}

    def _generator(self, scope_name="generator"):
        with tf.variable_scope(scope_name) as _:
            # 32 x 1 x 1 x 10
            y_reshape = tf.reshape(self.y, [-1, 1, 1, self.nb_targets])
            # 32 x 38
            h1 = tf.concat([self.z, self.y], 1)
            W1 = utils.weight_variable((h1.get_shape().as_list()[1], 7 * 7 * 1024), name="W1")
            # 32 x (7 x 7 x 1024)
            h1 = batch_norm(tf.matmul(h1, W1), is_training=self.is_training, scale=True)
            # 32 x 7 x 7 x 1024
            h2 = tf.reshape(h1, [-1, 7, 7, 1024])
            # 32 x 7 x 7 x 1034
            h2 = tf.concat([h2, y_reshape * tf.ones([-1, 7, 7, self.nb_targets])], 3)
            # 32 x 7 x 7 x 512
            h3 = batch_norm(conv2d_transpose(h2, 512, 5, 2), is_training=self.is_training, scale=True)
            # 32 x 14 x 14 x 256
            h4 = batch_norm(conv2d_transpose(h3, 256, 5, 2), is_training=self.is_training, scale=True)
            # 32 x 28 x 28 x 128
            h5 = batch_norm(conv2d_transpose(h4, 128, 5, 1), is_training=self.is_training, scale=True)
            # 32 x 28 x 28 x 1
            h6 = conv2d_transpose(h5, 1, 5, 1, activation_fn=tf.nn.tanh)
            return h6

    def _generator2(self, scope_name="generator"):
        with tf.variable_scope(scope_name) as _:
            x = fully_connected(
                self.z, 7 * 7 * 512, activation_fn=utils.lrelu, normalizer_fn=batch_norm)
            x = tf.reshape(x, (-1, 7, 7, 512))

            y_reshape = tf.reshape(self.y, [-1, 1, 1, self.nb_targets])
            x = tf.concat([x, y_reshape * tf.ones([tf.shape(self.y)[0], 7, 7, self.nb_targets])], 3)

            # 32 x 14 x 14 x 256
            x = conv2d_transpose(x, 256, 3, stride=2,
                                 activation_fn=tf.nn.relu, normalizer_fn=batch_norm, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
            x = tf.concat([x, y_reshape * tf.ones([tf.shape(self.y)[0], 14, 14, self.nb_targets])], 3)

            # 32 x 28 x 28 x 128
            x = conv2d_transpose(x, 128, 3, stride=2,
                                 activation_fn=tf.nn.relu, normalizer_fn=batch_norm, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
            # 32 x 28 x 28 x 64
            x = conv2d_transpose(x, 64, 3, stride=1,
                                 activation_fn=tf.nn.relu, normalizer_fn=batch_norm, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
            # 32 x 28 x 28 x 1
            x = conv2d_transpose(x, 1, 3, stride=1,
                                 activation_fn=tf.nn.tanh, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
            return x

    def _discriminator(self, image, reuse_scope, scope_name="discriminator"):
        with tf.variable_scope(scope_name, reuse=reuse_scope) as scope:
            if reuse_scope:
                scope.reuse_variables()
            num_outputs = 64
            x = conv2d(image, num_outputs=num_outputs, kernel_size=3,
                       stride=2, activation_fn=utils.lrelu)
            x = conv2d(x, num_outputs=num_outputs * 2, kernel_size=3,
                       stride=2, activation_fn=utils.lrelu, normalizer_fn=batch_norm)
            x = conv2d(x, num_outputs=num_outputs * 4, kernel_size=3,
                       stride=2, activation_fn=utils.lrelu, normalizer_fn=batch_norm)
            x = conv2d(x, num_outputs=num_outputs * 8, kernel_size=3,
                       stride=2, activation_fn=utils.lrelu, normalizer_fn=batch_norm)
            logit = fully_connected(tf.reshape(
                x, [-1, 2 * 2 * 512]), 1, activation_fn=None)
            return logit

    def build(self):
        self._create_placeholder()
        self.gen_images = self._generator2()
        self.x_reshape = tf.reshape(self.x, [-1, 28, 28, 1])

        self.features_real = self._discriminator(self.x_reshape, reuse_scope=False)
        self.features_fake = self._discriminator(self.gen_images, reuse_scope=True)
        self.optimize(self.features_real, self.features_fake)
        self.summary()

    def summary(self):
        trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variable:
            tf.summary.histogram(var.op.name, var)
        tf.summary.scalar("gen_loss", self.gen_loss)
        tf.summary.scalar("dis_loss", self.dis_loss)

        tf.summary.image("image_generated", self.gen_images, max_outputs=self.batch_size)
        tf.summary.image("true_image", self.x_reshape, max_outputs=self.batch_size)

        self.merged_summary_op = tf.summary.merge_all()

    def optimize(self, features_real, features_fake):
        self.dis_loss = tf.reduce_mean(features_fake - features_real)
        self.gen_loss = tf.reduce_mean(-features_fake)

        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gen_variables = [v for v in train_variables if v.name.startswith("generator")]
        print("Generator variable {}".format([v.op.name for v in self.gen_variables]))

        self.dis_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        print("Discriminator variable {}".format([v.op.name for v in self.dis_variables]))

        # batch norm operations (because batch_norm operations are not parent of train_dis, so we need
        # to tell tensorflow to still do the computation
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Batch norm variables {}".format([v.op.name for v in update_ops]))
        with tf.control_dependencies(update_ops):
            grads_dis = self.optimizer.compute_gradients(loss=self.dis_loss, var_list=self.dis_variables)
            self.train_dis = self.optimizer.apply_gradients(grads_dis)

            grads_gen = self.optimizer.compute_gradients(loss=self.gen_loss, var_list=self.gen_variables)
            self.train_gen = self.optimizer.apply_gradients(grads_gen)

    def _restore(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        last_saved_model = tf.train.latest_checkpoint('model')
        if last_saved_model is not None:
            saver.restore(self.sess, last_saved_model)
            print("Restoring model  {}".format(last_saved_model))
        return saver

    def train(self):
        summary_writer = tf.summary.FileWriter('logs', graph=self.sess.graph, flush_secs=60)
        clip_discriminator_var_op = utils.clip_op(self.dis_variables)
        saver = self._restore()
        for itr in tqdm(range(self.nb_iter)):
            if itr <= 25 or itr % 100 == 0:
                nb_critic_iteration = 100
            else:
                nb_critic_iteration = self.nb_critic_iter

            for _ in range(nb_critic_iteration):
                self.sess.run([self.train_dis], feed_dict=self._feed_dict(True))
                self.sess.run(clip_discriminator_var_op)

            self.sess.run(self.train_gen, feed_dict=self._feed_dict(True))
            if itr % 100 == 0:
                saver.save(self.sess, global_step=itr, save_path="model/model")
                print("Saving model")
                summary_str = self.sess.run(self.merged_summary_op, feed_dict=self._feed_dict(False))
                summary_writer.add_summary(summary_str, itr)

            if itr % 200 == 0:
                g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.dis_loss],
                                                       feed_dict=self._feed_dict(False))
                print(
                    "Iter {}: \n\tgenerator loss: {}, \n\tdiscriminator loss: {}".format(itr, g_loss_val,
                                                                                         d_loss_val))

    def visualize(self, numbers):
        nb_figures = len(numbers)
        self._restore()
        feed_dic = self._feed_dict(False, batch_size=len(numbers))
        y = np.zeros((nb_figures, 10))
        y[np.arange(nb_figures), numbers] = 1
        feed_dic[self.y] = y

        import matplotlib.pyplot as plt

        fig, (tuples) = plt.subplots(1, nb_figures)
        generated_images = self.sess.run(self.gen_images, feed_dict=feed_dic)
        generated_images = np.reshape(generated_images, (-1, 28, 28))

        for index in range(nb_figures):
            tuples[index].imshow(generated_images[index], cmap='gray')
            tuples[index].set_title('Number {}'.format(numbers[index]))
            tuples[index].axis('off')
        plt.axis('off')
        plt.show()


def main(argv):
    parser = ArgumentParser(description="WGAN_script")
    parser.add_argument("--train")
    parser.add_argument("--draw")
    args, _ = parser.parse_known_args()
    gan = GAN()
    gan.build()
    if args.train is not None:
        print("Train the model")
        gan.train()
    elif args.draw is not None:
        print("Draw a {}".format(args.draw))
        gan.visualize([int(item) for item in args.draw.split(',')])


if __name__ == '__main__':
    tf.app.run()
