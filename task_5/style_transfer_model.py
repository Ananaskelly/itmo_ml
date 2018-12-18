import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt


from task_5.utils import gram_matrix, load_and_process_img
from task_5.vgg_model import VGGModel
from task_5.utils import in_img


class StyleTransferModel:

    def __init__(self):
        pass

    def get_content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def get_style_loss(self, base_style, gram_target):
        gram_style = gram_matrix(base_style)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def get_feature_representations(self, model, content_path, style_path):

        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        style_outputs = model(style_image)
        content_outputs = model(content_image)

        style_features = [style_layer[0] for style_layer in style_outputs[:VGGModel.get_num_style_layers()]]
        content_features = [content_layer[0] for content_layer in content_outputs[VGGModel.get_num_style_layers():]]
        return style_features, content_features

    def compute_loss(self, model, loss_weights, init_image, gram_style_features, content_features):

        style_weight, content_weight = loss_weights

        model_outputs = model(init_image)

        style_output_features = model_outputs[:VGGModel.get_num_style_layers()]
        content_output_features = model_outputs[VGGModel.get_num_style_layers():]

        style_score = 0
        content_score = 0

        weight_per_style_layer = 1.0 / float(VGGModel.get_num_style_layers())
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)

        weight_per_content_layer = 1.0 / float(VGGModel.get_num_style_layers())
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def run_style_transfer(self,
                           content_path,
                           style_path,
                           num_iterations=1000,
                           content_weight=1e3,
                           style_weight=1e-2):

        model = VGGModel.get_layers()
        for layer in model.layers:
            layer.trainable = False

        style_features, content_features = self.get_feature_representations(model, content_path, style_path)
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        init_image = load_and_process_img(content_path)
        init_image = tfe.Variable(init_image, dtype=tf.float32)

        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        best_loss, best_img = float('inf'), None

        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []
        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            if loss < best_loss:
                best_loss = loss
                best_img = in_img(init_image.numpy())

            if i % display_interval == 0:
                start_time = time.time()

                plot_img = init_image.numpy()
                plot_img = in_img(plot_img)
                imgs.append(plot_img)
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))

        return best_img, best_loss
