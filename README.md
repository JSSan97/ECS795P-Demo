# ECS759-Demo
 
## How to train
Run main.py, changing arguments as necessary. E.g.
!python main.py --epochs=40 --model_name='ResNet101CBAM' --dataset='MNIST' --batch_size=64

## Plot Accuracy and Loss
Run experiment_plotter.py to view loss and accuracy during training. Change dataset that the model was trained with as necessary. E.g.
!python experiment_plotter.py --dataset='CIFAR10'. Edit global dict to change .npy paths containing the average loss and accuracy per epoch.

## Test Model on Input Images
Run experiment_images.py e.g. %run experiment_images.py --dataset='MNISTFashion' --batch_size=8 on google colab. 
Batch size refers to the number of image inputs.
