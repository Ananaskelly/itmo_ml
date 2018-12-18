from task_5.style_transfer_model import StyleTransferModel as STModel


import matplotlib.pyplot as plt

content_path = '../data/style_transfer/content.jpg'
style_path = '..data/style_transfer/style.jpg'
st_model = STModel()
best, best_loss = st_model.run_style_transfer(content_path, style_path, num_iterations=10)

plt.imshow(best)
