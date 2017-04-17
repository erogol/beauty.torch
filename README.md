<h1>beauty.torch</h1>

This project serves a deep learning model scoring selfie images between 1 to 10 based on image
and face attributes. You can learn the technical details of this project from this [blog post](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/). Use [resnet.torch] (https://github.com/erogol/resnet.torch), if you plan to follow all the training pipeline described on the post. 

Given image is processed as follows;  

1. Detect face.
2. Find landmarks
3. Rotate image to align face.
4. Fill gaps with constant pixel value.
5. Send into scoring model.

For an example use check notebook ```ExampleUse.ipynb```  

<h2>Requirements</h2>
Main requirement is [Torch](http://torch.ch/docs/getting-started.html) computing framework.  

<h4>Models</h4>
[dlib face model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) - place under ```utils/```  
[beauty model](https://www.dropbox.com/s/yezp73uxqnd86e6/model_best.t7) - GPU model (use ```utils/convert2cpu.lua``` for setting it for CPU) place under ```trained/```  
[optimstate](https://www.dropbox.com/s/yezp73uxqnd86e6/model_best.t7) - if you like to fine-tune the model. 

<h4>Python</h4>  
dlib ```sudo pip install dlib``` - face and landmark detection)  
lutorpy ```sudo pip install lutorpy``` - using torch model on python  
skimage ```sudo pip install skimage``` - image processing  
cv2 ```sudo pip install cv2``` - OpenCV python module  

<h2>What you have here useful</h2>
* Face alignment code in ```utils/img_processing.py```.
* A template for porting Torch models to python in ```utils/Classifier.py```.
* The model itself

<h2>Examples</h2>
Attention of the trained model.  
![alt test](https://raw.githubusercontent.com/erogol/beauty.torch/master/exps/pitt.png "Model attention")

Sorting A. Lima images from Google Search.  
![alt tag](https://raw.githubusercontent.com/erogol/beauty.torch/master/exps/out.gif)
