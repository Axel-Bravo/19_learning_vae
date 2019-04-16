import PIL
import glob
import imageio
import matplotlib.pyplot as plt


def create_gif(images_folder: str, gif_name: str) -> None:
    """
    Creates a gif from a given folder of images
    :param images_folder: folder containing images with the pattern 'image*.png'
    :param gif_name: name of the gif file
    :return: None, saves gif on disk
    """

    with imageio.get_writer(images_folder + gif_name, mode='I') as writer:
        filenames = glob.glob(images_folder + 'image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def display_image(images_folder: str, epoch_no: int) -> PIL.Image:
    """
    Display the selected epoch's image of a given folder
    :param images_folder: folder containing images with the pattern 'image*.png'
    :param epoch_no: number of epoch selected to be seen
    :return: selected image
    """
    return PIL.Image.open(images_folder + 'image_at_epoch_{:04d}.png'.format(epoch_no))


def generate_and_save_images(model, epoch, test_input):
    """
    Generate and save image, used to see the evolution of the neural network
    :param model: neural network model
    :param epoch: epoch number the model is generating
    :param test_input: random noise introduced to the generative model
    :return: generated image
    """
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

