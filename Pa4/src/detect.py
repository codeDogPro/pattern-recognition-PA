from model import ObjectDetectionModel
from gui import display, select_picture, process_img


def main():
    model = ObjectDetectionModel()
    model.eval()
    while True:
        # read image (give interface to user to pick the pictures in their computer)
        image = select_picture()

        # put it into model and get all bboxs
        bboxs = model(image)

        # process this image with these bboxs
        processed_img = process_img(image=image, bboxs=bboxs)

        # display the image(we better display both raw and processed image)
        display(processed_img)

        del image, processed_img


if __name__ == '__main__':
    main()
