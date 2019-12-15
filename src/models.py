from __future__ import print_function

import os
import time
import random
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from .networks import Generator, Discriminator
from .ops import pixelwise_accuracy, preprocess, postprocess
from .ops import COLORSPACE_RGB, COLORSPACE_LAB
from .dataset import Places365Dataset, Cifar10Dataset, TestDataset
from .utils import stitch_images, turing_test, imshow, imsave, create_dir, visualize, Progbar


class BaseModel:
    def __init__(self, sess, options):
        self.sess = sess
        self.options = options
        self.name = options.name
        self.samples_dir = os.path.join(options.checkpoints_path, 'samples')
        self.test_log_file = os.path.join(options.checkpoints_path, 'log_test.dat')
        self.train_log_file = os.path.join(options.checkpoints_path, 'log_train.dat')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.dataset_train = self.create_dataset(True)
        self.dataset_val = self.create_dataset(False)
        self.sample_generator = self.dataset_val.generator(options.sample_size, True)
        self.iteration = 0
        self.epoch = 0
        self.is_built = False

    def train(self):
        total = len(self.dataset_train)

        for epoch in range(self.options.epochs):
            lr_rate = self.sess.run(self.learning_rate)

            print('Training epoch: %d' % (epoch + 1) + " - learning rate: " + str(lr_rate))

            self.epoch = epoch + 1
            self.iteration = 0

            generator = self.dataset_train.generator(self.options.batch_size)
            progbar = Progbar(total, width=25, stateful_metrics=['epoch', 'iter', 'step'])

            for input_rgb in generator:
                feed_dic = {self.input_rgb: input_rgb}

                self.iteration = self.iteration + 1
                self.sess.run([self.dis_train], feed_dict=feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)

                lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, lossG_color, acc, step = self.eval_outputs(feed_dic=feed_dic)

                progbar.add(len(input_rgb), values=[
                    ("epoch", epoch + 1),
                    ("iter", self.iteration),
                    ("step", step),
                    ("D loss", lossD),
                    ("D fake", lossD_fake),
                    ("D real", lossD_real),
                    ("G loss", lossG),
                    ("G L1", lossG_l1),
                    ("G gan", lossG_gan),
                    ("G color loss", lossG_color),
                    ("accuracy", acc)
                ])

                # log model at checkpoints
                if self.options.log and step % self.options.log_interval == 0:
                    with open(self.train_log_file, 'a') as f:
                        f.write('%d %d %f %f %f %f %f %f %f\n' % (self.epoch, step, lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, acc))

                    if self.options.visualize:
                        visualize(self.train_log_file, self.test_log_file, self.options.visualize_window, self.name)

                # sample model at checkpoints
                if self.options.sample and step % self.options.sample_interval == 0:
                    self.sample(show=False)

                # validate model at checkpoints
                if self.options.validate and self.options.validate_interval > 0 and step % self.options.validate_interval == 0:
                    self.validate()

                # save model at checkpoints
                if self.options.save and step % self.options.save_interval == 0:
                    self.save()

            if self.options.validate:
                self.validate()

    def validate(self):
        print('\n\nValidating epoch: %d' % self.epoch)
        total = len(self.dataset_val)

        val_generator = self.dataset_val.generator(self.options.batch_size)
        progbar = Progbar(total, width=25)

        for input_rgb in val_generator:
            feed_dic = {self.input_rgb: input_rgb}

            self.sess.run([self.dis_loss, self.gen_loss, self.accuracy], feed_dict=feed_dic)

            lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, lossG_color, acc, step = self.eval_outputs(feed_dic=feed_dic)

            progbar.add(len(input_rgb), values=[
                ("D loss", lossD),
                ("D fake", lossD_fake),
                ("D real", lossD_real),
                ("G loss", lossG),
                ("G L1", lossG_l1),
                ("G gan", lossG_gan),
                ("G color loss", lossG_color),
                ("accuracy", acc)
            ])

        print('\n')


    def evaluate_color(self):

        # This code runs through test set and evaluates the RGB color values in each image
        val_generator = self.dataset_val.generator(self.options.batch_size)

        rgbs = []
        for input_rgb in val_generator:
            feed_dic = {self.input_rgb: input_rgb}
            outputs = self.sess.run(self.output_rgb, feed_dict=feed_dic)
            rgb = np.mean(outputs * 255, axis=(0, 1, 2))
            rgbs.append(rgb)


        print("Avg RGB: " + str(np.mean(rgbs, axis=0)))

    def infer_and_save(self, num=500):

        # This method will run and submit
        val_generator = self.dataset_val.generator(self.options.batch_size)
        outputs_path = create_dir(self.options.test_output or (self.options.checkpoints_path + '/output'))

        k = 0
        for input_rgb in val_generator:
            feed_dic = {self.input_rgb: input_rgb}
            outputs = self.sess.run(self.sampler, feed_dict=feed_dic)
            outputs = postprocess(tf.convert_to_tensor(outputs), colorspace_in=self.options.color_space,
                                  colorspace_out=COLORSPACE_RGB).eval() * 255
            for j in range(len(outputs)):
                imsave(outputs[j], os.path.join(outputs_path, '{}.jpg'.format(k)))
                k += 1
                if k == num:
                    break


    def sample(self, show=True):
        input_rgb = next(self.sample_generator)
        feed_dic = {self.input_rgb: input_rgb}

        step, rate = self.sess.run([self.global_step, self.learning_rate])
        fake_image, input_gray = self.sess.run([self.sampler, self.input_gray], feed_dict=feed_dic)
        fake_image = postprocess(tf.convert_to_tensor(fake_image), colorspace_in=self.options.color_space, colorspace_out=COLORSPACE_RGB)
        img = stitch_images(input_gray, input_rgb, fake_image.eval())

        create_dir(self.samples_dir)
        sample = self.options.dataset + "_" + str(step).zfill(5) + ".png"

        if show:
            imshow(np.array(img), self.name)
        else:
            print('\nsaving sample ' + sample + ' - learning rate: ' + str(rate))
            img.save(os.path.join(self.samples_dir, sample))


    def turing_test(self):
        # assumes 4 subdirs in this directory, 'real', 'baseline', 'cool', 'warm'
        image_list = []
        for root, subdirs, files in os.walk(self.options.turing_test_directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_list.append(os.path.join(root, file))
        random.shuffle(image_list)
        results = {'real': [0, 0], 'baseline': [0, 0], 'cool': [0, 0], 'warm': [0, 0]}
        for image_file in image_list:
            image_file = image_file.replace("\\", '/')
            fooled = turing_test(image_file, delay=self.options.turing_test_delay)
            image_type = image_file.split('/')[-2]
            results[image_type][0] += fooled
            results[image_type][1] += 1

        for entry in results:
            print('Percentage of {} thought to be real: {}'.format(entry, results[entry][0] / results[entry][1] * 100))


    def build(self):
        if self.is_built:
            return

        self.is_built = True

        gen_factory = self.create_generator()
        dis_factory = self.create_discriminator()
        smoothing = 0.9 if self.options.label_smoothing else 1
        seed = self.options.seed
        kernel = 4

        # model input placeholder: RGB imaege
        self.input_rgb = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='input_rgb')


        # model input after preprocessing: LAB image
        self.input_color = preprocess(self.input_rgb, colorspace_in=COLORSPACE_RGB, colorspace_out=self.options.color_space)

        # test mode: model input is a graycale placeholder
        # if self.options.mode == 1:
        #     self.input_gray = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='input_gray')

        # train/turing-test we extract grayscale image from color image
        # else:
        self.input_gray = tf.image.rgb_to_grayscale(self.input_rgb)

        gen = gen_factory.create(self.input_gray, kernel, seed)
        # gen = tf.concat([self.input_color[:, :, :, :1], gen[:, :, :, 1:]], axis=-1)

        dis_real = dis_factory.create(tf.concat([self.input_gray, self.input_color], 3), kernel, seed)
        dis_fake = dis_factory.create(tf.concat([self.input_gray, gen], 3), kernel, seed, reuse_variables=True)

        gen_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.ones_like(dis_fake))
        dis_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_real) * smoothing)
        dis_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake))

        self.dis_loss_real = tf.reduce_mean(dis_real_ce)
        self.dis_loss_fake = tf.reduce_mean(dis_fake_ce)
        self.dis_loss = tf.reduce_mean(dis_real_ce + dis_fake_ce)

        self.gen_loss_gan = tf.reduce_mean(gen_ce)
        self.gen_loss_l1 = tf.reduce_mean(tf.abs(self.input_color - gen)) * self.options.l1_weight

        color_sceheme_color = tf.constant(self.options.scheme_color)
        self.color_scheme_loss = tf.reduce_mean(tf.abs(color_sceheme_color[1:] - gen[:, :, :, 1:])) * self.options.color_weight # gen = output of the generator (array of h * w * 3)
        self.gen_loss = self.gen_loss_gan + self.gen_loss_l1 + self.color_scheme_loss # final objective function

        self.sampler = tf.identity(gen_factory.create(self.input_gray, kernel, seed, reuse_variables=True), name='output')

        self.output_rgb = postprocess(self.sampler, colorspace_in=self.options.color_space, colorspace_out=COLORSPACE_RGB)

        self.accuracy = pixelwise_accuracy(self.input_color, gen, self.options.color_space, self.options.acc_thresh)
        self.learning_rate = tf.constant(self.options.lr)

        # learning rate decay
        if self.options.lr_decay and self.options.lr_decay_rate > 0:
            self.learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
                learning_rate=self.options.lr,
                global_step=self.global_step,
                decay_steps=self.options.lr_decay_steps,
                decay_rate=self.options.lr_decay_rate))

        # generator optimizaer
        self.gen_train = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.options.beta1
        ).minimize(self.gen_loss, var_list=gen_factory.var_list)

        # discriminator optimizaer
        self.dis_train = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate / 10,
            beta1=self.options.beta1
        ).minimize(self.dis_loss, var_list=dis_factory.var_list, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.options.checkpoints_path)
        if ckpt is not None:
            print('loading model...\n')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.options.checkpoints_path, ckpt_name))
            return True

        return False

    def save(self):
        print('saving model...\n')
        self.saver.save(self.sess, os.path.join(self.options.checkpoints_path, 'CGAN_' + self.options.dataset), write_meta_graph=False)

    def eval_outputs(self, feed_dic):
        '''
        evaluates the loss and accuracy
        returns (D loss, D_fake loss, D_real loss, G loss, G_L1 loss, G_gan loss, accuracy, step)
        '''
        lossD_fake = self.dis_loss_fake.eval(feed_dict=feed_dic)
        lossD_real = self.dis_loss_real.eval(feed_dict=feed_dic)
        lossD = self.dis_loss.eval(feed_dict=feed_dic)

        lossG_l1 = self.gen_loss_l1.eval(feed_dict=feed_dic)
        lossG_gan = self.gen_loss_gan.eval(feed_dict=feed_dic)
        lossG_color = self.color_scheme_loss.eval(feed_dict=feed_dic)
        lossG = lossG_l1 + lossG_gan + lossG_color

        acc = self.accuracy.eval(feed_dict=feed_dic)
        step = self.sess.run(self.global_step)

        return lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, lossG_color, acc, step

    @abstractmethod
    def create_generator(self):
        raise NotImplementedError

    @abstractmethod
    def create_discriminator(self):
        raise NotImplementedError

    @abstractmethod
    def create_dataset(self, training, turing):
        raise NotImplementedError


class Cifar10Model(BaseModel):
    def __init__(self, sess, options):
        super(Cifar10Model, self).__init__(sess, options)

    def create_generator(self):
        kernels_gen_encoder = [
            (64, 1, 0),     # [batch, 32, 32, ch] => [batch, 32, 32, 64]
            (128, 2, 0),    # [batch, 32, 32, 64] => [batch, 16, 16, 128]
            (256, 2, 0),    # [batch, 16, 16, 128] => [batch, 8, 8, 256]
            (512, 2, 0),    # [batch, 8, 8, 256] => [batch, 4, 4, 512]
            (512, 2, 0),    # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        ]

        kernels_gen_decoder = [
            (512, 2, 0.5),  # [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (256, 2, 0.5),  # [batch, 4, 4, 512] => [batch, 8, 8, 256]
            (128, 2, 0),    # [batch, 8, 8, 256] => [batch, 16, 16, 128]
            (64, 2, 0),     # [batch, 16, 16, 128] => [batch, 32, 32, 64]
        ]

        return Generator('gen', kernels_gen_encoder, kernels_gen_decoder, training=self.options.training)

    def create_discriminator(self):
        kernels_dis = [
            (64, 2, 0),     # [batch, 32, 32, ch] => [batch, 16, 16, 64]
            (128, 2, 0),    # [batch, 16, 16, 64] => [batch, 8, 8, 128]
            (256, 2, 0),    # [batch, 8, 8, 128] => [batch, 4, 4, 256]
            (512, 1, 0),    # [batch, 4, 4, 256] => [batch, 4, 4, 512]
        ]

        return Discriminator('dis', kernels_dis, training=self.options.training)

    def create_dataset(self, training=True, turing=False):
        return Cifar10Dataset(
            path=self.options.dataset_path,
            training=training,
            augment=self.options.augment,
            turing=turing)


class Places365Model(BaseModel):
    def __init__(self, sess, options):
        super(Places365Model, self).__init__(sess, options)

    def create_generator(self):
        kernels_gen_encoder = [
            (64, 1, 0),     # [batch, 256, 256, ch] => [batch, 256, 256, 64]
            (64, 2, 0),     # [batch, 256, 256, 64] => [batch, 128, 128, 64]
            (128, 2, 0),    # [batch, 128, 128, 64] => [batch, 64, 64, 128]
            (256, 2, 0),    # [batch, 64, 64, 128] => [batch, 32, 32, 256]
            (512, 2, 0),    # [batch, 32, 32, 256] => [batch, 16, 16, 512]
            (512, 2, 0),    # [batch, 16, 16, 512] => [batch, 8, 8, 512]
            (512, 2, 0),    # [batch, 8, 8, 512] => [batch, 4, 4, 512]
            (512, 2, 0)     # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        ]

        kernels_gen_decoder = [
            (512, 2, 0),    # [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (512, 2, 0),    # [batch, 4, 4, 512] => [batch, 8, 8, 512]
            (512, 2, 0),    # [batch, 8, 8, 512] => [batch, 16, 16, 512]
            (256, 2, 0),    # [batch, 16, 16, 512] => [batch, 32, 32, 256]
            (128, 2, 0),    # [batch, 32, 32, 256] => [batch, 64, 64, 128]
            (64, 2, 0),     # [batch, 64, 64, 128] => [batch, 128, 128, 64]
            (64, 2, 0)      # [batch, 128, 128, 64] => [batch, 256, 256, 64]
        ]

        return Generator('gen', kernels_gen_encoder, kernels_gen_decoder, training=self.options.training)

    def create_discriminator(self):
        kernels_dis = [
            (64, 2, 0),     # [batch, 256, 256, ch] => [batch, 128, 128, 64]
            (128, 2, 0),    # [batch, 128, 128, 64] => [batch, 64, 64, 128]
            (256, 2, 0),    # [batch, 64, 64, 128] => [batch, 32, 32, 256]
            (512, 1, 0),    # [batch, 32, 32, 256] => [batch, 32, 32, 512]
        ]

        return Discriminator('dis', kernels_dis, training=self.options.training)

    def create_dataset(self, training=True, turing=False):
        return Places365Dataset(
            path=self.options.dataset_path,
            training=training,
            augment=self.options.augment,
            turing=turing)
