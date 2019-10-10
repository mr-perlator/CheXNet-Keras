import cv2
import numpy as np
import os
import pandas as pd
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from tensorflow.keras import backend as kb


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def create_cam(df_g, output_dir, image_source_dir, model, generator, class_names, targets=None):
    """
    Create a CAM overlay image for the input image

    :param targets: producs CAMs only for these labels. if left empty, CAMs are generated for all labels
    :param df_g: pandas.DataFrame with file paths
    :param output_dir: str
    :param image_source_dir: str
    :param model: keras model
    :param generator: generator.AugmentedImageSequence
    :param class_names: list of str
    """
    file_name = df_g["Path"]
    print(f"**process image: {file_name}**")

    # draw cam with labels
    img_ori = cv2.imread(filename=os.path.join(image_source_dir, file_name))
    """
    label = df_g["label"]
    if label == "Infiltrate":
        label = "Infiltration"
    """
    # label = "Lung Opacity"
    if targets is None:
        targets = class_names

    img_transformed = generator.load_image(file_name)

    # CAM overlay
    # Get the 1024 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    # print(class_weights.shape)
    # print(index)
    # print(class_weights[..., index].shape)
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_transformed])])
    conv_outputs = conv_outputs[0, :, :, :]

    for label in targets:
        # print(f"process label: {label}")
        index = class_names.index(label)
        output_file = f"{file_name.split('/')[-1].split('.')[-2]}_{label}.{file_name.split('.')[-1]}"
        output_path = os.path.join(output_dir, output_file)

        # Create the class activation map.
        cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
        for i, w in enumerate(class_weights[..., index]):
            cam += w * conv_outputs[:, :, i]
        # print(f"predictions: {predictions}")
        cam /= np.max(cam)
        # cam[np.where(cam < 0.2)] = 0
        cam = cv2.resize(cam, img_ori.shape[:2][::-1])  # flip image dimensions, see https://stackoverflow.com/questions/21248245/opencv-image-resize-flips-dimensions

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap * 0.5 + img_ori

        # add label & predicted probability

        cv2.putText(img, text=label + f": {str(predictions[...,index])}", org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=1)
        cv2.imwrite(output_path, img)


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # CAM config
    # cam_list_file = cp["CAM"].get("cam_list_file")
    cam_folder = cp["CAM"].get("cam_folder")
    use_best_weights = cp["CAM"].getboolean("use_best_weights")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path

    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    # print(model.summary())

    print("read contents of cam folder")
    cam_files = [f for f in os.listdir(cam_folder) if os.path.isfile(os.path.join(cam_folder, f))]
    df_images = pd.DataFrame(cam_files)
    df_images.columns = ["Path"]

    print("create a generator for loading transformed images")
    cam_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "valid.csv"),  # variable must be passed, but is not used for CAMs
        class_names=class_names,
        source_image_dir=cam_folder,
        batch_size=1,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=1,
        shuffle_on_epoch_end=False,
    )

    image_output_dir = os.path.join(output_dir, "cam_output")
    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)

    print("create CAM")
    df_images.apply(
        lambda g: create_cam(
            df_g=g,
            output_dir=image_output_dir,
            image_source_dir=cam_folder,
            model=model,
            generator=cam_sequence,
            class_names=class_names,
            # targets=["Lung Lesion"]
        ),
        axis=1,
    )


if __name__ == "__main__":
    main()
