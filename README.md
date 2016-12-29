<h3>beauty.torch</h3>

This project serves a deep learning model scoring selfie images between 1 to 10 based on image
and face attributes. It is developed based on [resnet.torch] (https://github.com/erogol/resnet.torch).

This repository also includes some additional codes to preprocess images
before serving into the model. It basically follows three important steps.

1. Detect face.
2. Find landmarks
3. Rotate image to align face.
4. Fill gaps with constant pixel value.
5. Send into scoring model.

<h5>Requirements</h5>
models ---
[dlib face model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) - place under ```utils/```
[beauty model](https://www.dropbox.com/s/yezp73uxqnd86e6/model_best.t7) - place under ```trained/```
[optimstate](https://www.dropbox.com/s/yezp73uxqnd86e6/model_best.t7)

python ---
dlib (face and landmark detection)
lutorpy (using torch model on python)
skimage (image processing)

<h5>What you have here useful</h5>
* Face alignment code in ```utils/img_processing.py```.
* A template for porting Torch models to python in ```utils/Classifier.py```.
* The model itself :)

<h5>Examples</h5>
Attention of the trained model.
![alt tag](https://raw.githubusercontent.com/erogol/beauty.torch/master/exps/pitt.png)

Sorting A. Lima images from Google Search.
![alt tag](https://raw.githubusercontent.com/erogol/beauty.torch/master/exps/out.gif)
