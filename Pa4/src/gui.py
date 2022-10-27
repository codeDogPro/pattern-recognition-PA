# display(), precess_img(), select_picture() are exposed to user,
# you can add any function to implement these three interfaces.

def select_picture():
    """
    Let the user select a picture to be detected.
    :return: raw image without any process
    """
    print('Not implemented')
    assert 0


def process_img(image, bboxs: list[(int, int, int, int)]):
    """
    To process the image, drawing the bboxs given;
    :param image: the raw image;
    :param bboxs: all the detected boundingboxs. format: list[(x, y, w, h)]
    :return: the precessed image(not a tensor)
    """
    print('Not implemented')
    assert 0


def display(image):
    """
    Display the image;
    :param image: raw image or processed image
    """
    print('Not implemented')
    assert 0
