# Hand_Sign_Recognition_Classifer_Advanced_ML
Hand Sign Recognition Classifier implemented in PyTorch that predicts the correct label given an image of a hand doing a sign representing 0, 1, 2, 3, 4 or 5. The dataset can be obtained fromn this link : https://drive.google.com/file/d/1rzq38ngenOve93UTjQeE2M9REk3g8j7L/view
The dataset contains 64x64 sized images that are named following {label}_IMG_{id}.jpg where the label is in [0, 5]. The model is trained for 10 epochs in a batch size of 32.
The following regularization strategies are tried:
1. L2-norm penalty, 
2. early stopping, 
3. data augmentation, 
4. dropout
and the following optimization techniques are tried: 
1. GD, 
2. SGD (1 example, mini-batch), 
3. AdaGrad, 
4. RMSProp, 
5. RMSProp with momentum, 
6. Adam, 
7. Batch Normalization
The loss and accuracy is plotted per epoch.
