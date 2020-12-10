# Learning-without-Forgetting-using-Pytorch
This is the Pytorch implementation of LwF

In my experiment, the baseline is Alexnet from Pytorch whose top1 accuracy is 56.518 and top5 accuracy is 79.070% (I only use top1 accuracy in my experiment following). I use CUB data set. The dataset.py refered to: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks

If you want to find the paper, you can click ：https://ieeexplore.ieee.org/document/8107520

I suggest that 'lr' should be set as 0.001 and 'alpha' should be set less than 0.3. 
If 'alpha' is too large, the accuracy of old classes will decrease very quickly. But if it is too small, the accuracy of new classes will be lower.
I set T as 2, I didn't try other values so I don't know whether there will be other values that lead to higher performance. If you want to know detail of this super-parameter, you can read this paper: https://arxiv.org/pdf/1503.02531.pdf。 This paper is the source of 'Knowledge Distillation'.
There are some result:

number of new classes      alpha      accuracy of old 1000 classes      accuracy of new classes
      50                    0.1                   55.120%                       47.948%
      50                    0.3                   51.512%                       54.664%
      100                   0.1                   54.208%                       39.525%
      100                   0.3                   50.002%                       48.092%
      150                   0.1                   53.284%                       36.573%
      150                   0.3                   49.116%                       45.284%
      200                   0.1                   53.370%                       34.413%
      200                   0.3                   49.342%                       44.492%
      
If I only add one class, the accuracy of the new class will be very high with a litle decrease of accuracy of old 1000 classes.
I tried this in CUB data set and the ave-acc of new 200 classes is 80.98% and the accuracy of old 1000 classes is 55.035% (-1.465%).

I holp you can chech your curves after implementing this code. You will find that, at the begining, the accuracy of new classes will be always 0 for some epoches but the accuracy of old classes will decrease quickly. I don't know why. If you have any interesting finding, please contact me.
