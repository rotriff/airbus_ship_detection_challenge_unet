# airbus_ship_detection_challenge_unet
Locate ships in images using unet architecture

## Solution summary
Solution is based on Unet neural network architechture. Masks are represented as run-length encoded string for each individual ship in the image, so first we define function for run-length encoding (which we  will use in future), decoding and also function for representation of single mask for all ships in picture. 

For explotarorary purpose we modify dataframe to see disbalance in ships number per image, as there are too many of emty pictures we undersample dataset (just delete empty pictures). Then we create image generator which decodes masks and produce training and validation datasets. To achieve higher performance we apply data augmentation applying zoom, shift, flip etc. For this purpose we also create datagenerator.

Network is conventional unet best fitted for semantic segmentation with some dropout layers to prevent overfittiong and padding, activation is the most popular ReLu function. Also we create custom loss and score functions which are just Dice score or F1, optimizing loss function using adam, metrics binary accuracy and f1.

To predict allocation of ships on test images we create two functions, first make raw prediction and return mask and sekond run-length encodes it, so output dataset coud be saved as two column frame with ImageID and run-length encoded string corresponding to mask.

As conclusion it should be mentioned that model has a room to grow, as it has notable imperfections, though has decent scores: loss: -0.5898 - dice_coef: 0.5906 - binary_accuracy: 0.9782 - val_loss: -0.2898 - val_dice_coef: 0.3163 - val_binary_accuracy: 0.8258

Cropped test dataset is present on GitHub for presentation and testing.
https://github.com/shellyguns/airbus_ship_detection_challenge_unet

Full dataset and completed task with exploratory analysis on Kaggle.
https://www.kaggle.com/code/vovataran/airbus-challenge-unet

## Packages used
- [Tensorflow](https://www.tensorflow.org/)
- [Scikit-learn](http://scikit-learn.org)
- [Scikit-image](https://scikit-image.org/)
- [Keras](https://keras.io/)
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://tqdm.github.io/)


by Vova Taran
