# airbus_ship_detection_challenge_unet
Locate ships in images using unet architecture

## Solution summary
Solution is based on Unet neural network architechture. Masks are represented as run-length encoded string for each individual ship in the image, so first we define function for run-length encoding (which we  will use in future), decoding and also function for representation of single mask for all ships in picture. 

For explotarorary purpose we modify dataframe to see disbalance in ships number per image, as there are too many of emty pictures we undersample dataset (just delete empty pictures). Then we create image generator which decodes masks and produce training and validation datasets. To achieve higher performance we apply data augmentation applying zoom, shift, flip etc. For this purpose we also create datagenerator.

Network is conventional unet best fitted for semantic segmentation with some dropout layers to prevent overfittiong and padding, activation is the most popular ReLu function. Also we create custom loss and score functions which are just Dice score or F1, optimizing loss function using adam, metrics binary accuracy and f1.