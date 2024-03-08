# AutoEncoder for anomalies detections.

Based on the retrieved information on the Data Analysis notebook, we know this:
- The graph has 46564 nodes and 36624 edges.
- The graph contains **42019 licit** transactions and **4545 illicit** transactions.
- Thanks to PCA, we have reduced 169 original features to 73.
- After some metrics computation, we have other 10 features, mainly based on network metrics.

> The most important aspect to take into account is the unbalanced dataset we are working with. A ***90.24%*** of the dataset is comprehended by licit transactions. This is, less than a ***10%*** forms the total of illicit transactions. 
> We could reduce the size of the licit nodes by random sampling, making the data balanced. If we do this, we will be able to train a classifier, for example, but it will probably have a poor performance, since the dataset will be quite small and not representative at all *(specially if we remove 80% of the original dataset)*.

#### Therefore, my idea for this approach is the following:
Based on the unbalanced data and without having any intention of reducing its size, we could train an autoencoder just with licit samples. Since we have an enormous amount of licit nodes, the autoencoder may this way learn to represent these kind of information really well. 

Following this idea, if our model knows how to reconstruct licit nodes but we give to it an illicit node to reconstruct, some anomaly will occur. Therefore, we have to be able to detect this anomaly coming from our autoencoder. 

**Two main options are possible (*maybe there are other to take into account*):**

1. Train a classifier that learns to differentiate between reconstructed licit nodes and reconstructed illicit nodes.

2. Analyizing the results with the computation of a ROC curve and deciding a threshold of permeability, classifying this way the two different classes of our problem.
    
    2.1. Thanks to this option, it may be possible to give a degree of belief of our model in relation with the result. Whether our model is sure that the result is correct or it is not.

#### There are many different ideas that I would like to implement:
First, I'd like to check a boxplot with the mean of all PCA variables. This way, I may be able to detect outliers. Once I have detected them, I will choose either to use them on the AE training or not, and I will have to choose if I want to use them for the ROC curve training too.

Secondly, will have to split the train, test and validation data. However, this has some complexity: I have to save a very big part of the data to train the AutoEncoder model. With the remaining data, which will be balanced between licit and illicit samples, I will have to do a split in train and test. 

# Use reconstruction loss (ROC)

\
**Wish me luck!**

***Pol***