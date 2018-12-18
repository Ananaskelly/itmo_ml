import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from task_5.style_transfer_model import StyleTransferModel as STModel
from task_5.utils import load_img_to_show


if __name__ == '__main__':
    tf.enable_eager_execution()

    content_path = './data/style_transfer/content.jpg'
    style_path = './data/style_transfer/style.jpg'
    st_model = STModel()
    best_pic, best_loss = st_model.run_style_transfer(content_path, style_path, num_iterations=700)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(load_img_to_show(content_path))
    plt.axis('off')
    fig.add_subplot(1, 3, 2)
    plt.imshow(load_img_to_show(style_path))
    plt.axis('off')
    fig.add_subplot(1, 3, 3)
    plt.imshow(best_pic)
    plt.axis('off')

    plt.savefig('result_mix.png')
    plt.imsave('result_solo.jpg', best_pic)
