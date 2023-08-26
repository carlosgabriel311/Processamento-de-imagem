from matplotlib import pyplot as plt


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input image', 'True mask', 'Predicted mask']

  for i in range(len(display_list)):
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(display_list[0])
    plt.title('Imagem de Teste')
    plt.subplot(132)
    plt.imshow(display_list[1].squeeze(), cmap='gray')
    plt.title('Máscara Real')
    plt.subplot(133)
    plt.imshow(display_list[2].squeeze(), cmap='gray')
    plt.title('Máscara Prevista')
    plt.savefig('teste.png')