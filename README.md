# CheXPert-Keras
This project is a tool to build CheXNet-like models, written in Keras.
Forked, based on original repository by Bruce Chou (brucechou1983@gmail.com).
Several changes were made to tailor the code to the CheXPert dataset. Updated for Tensorflow 2.0.
Now uses tf.keras. For now no image augmentation.

## What is [CheXPert](https://arxiv.org/pdf/1901.07031.pdf)?
ChexNet is a deep learning algorithm that can detect and localize 14 kinds of diseases from chest X-ray images. As described in the paper, a 121-layer densely connected convolutional neural network is trained on CheXPert dataset, which contains 224,316 frontal and side view X-ray images from 65,240 unique patients. If you are new to this project, [Luke Oakden-Rayner's post](https://lukeoakdenrayner.wordpress.com/2019/02/25/half-a-million-x-rays-first-impressions-of-the-stanford-and-mit-chest-x-ray-datasets/) is highly recommended.

## In this project, you can
1. Train/test a **baseline model** by following the quickstart.
2. Run class activation mapping to see the localization of your model.
3. Modify `multiply` parameter in `config.ini` or design your own class weighting to see if you can get better performance.
4. Modify `weights.py` to customize your weights in loss function.
5. Every time you do a new experiment, make sure you modify `output_dir` in `config.ini` otherwise previous training results might be overwritten. For more options check the parameter description in `config.ini`.

## Quickstart
1. Download CheXPert dataset from [Stanford ML group](https://stanfordmlgroup.github.io/competitions/chexpert/). Put it under `./data` folder. Use CheXpert-v1.0-small
2. Create & source a new virtualenv. Python >= **3.6** is required.
3. Install dependencies by running `pip3 install -r requirements.txt`.
4. Copy sample_config.ini to config.ini, you may customize `batch_size` and training parameters here. Make sure config.ini is configured before you run training or testing
5. Run `python train.py` to train a new model. If you want to run the training using multiple GPUs, just prepend `CUDA_VISIBLE_DEVICES=0,1,...` to restrict the GPU devices. `nvidia-smi` command will be helpful if you don't know which device are available.
6. Run `python test.py` to evaluate your model on the test set.
7. Create a `./data/cam` folder to specify images for custom class activation maps or change config.ini accordingly
7. Run `python cam.py` to generate images with class activation mapping overlay. CAM images will be placed under the output folder.

## Author
Aleksej Perlov (aleksej.perlov@gmail.com), after fork from Bruce Chou (brucechou1983@gmail.com)

## License
MIT
