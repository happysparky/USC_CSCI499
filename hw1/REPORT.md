# Instructions 
Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set.

## Note
I didn't use model.py, all my code is in train.py and utils.py

# To run this code:  
Make sure all relevant packages are installed (detailed at the top of train.py and utils.py).
Make sure you're using python3 (I use 3.9.12).
Ensure all required flags are set when typing command to run. In addition to the default flags, I added flags for
the embedding dimension (--emb-dim) and learning rate (--learning_rate). However, these have default values (128 and 0.001, respectively), values don't necessarily have to be specified for them. 

## Example command (no stemming):  
python train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=experiments/lstm.png --batch_size=128 --num_epochs=101 --val_every=5 --force_cpu --emb_dim=128 --learning_rate=0.001

## Example command (with stemming):
python train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=experiments/lstm.png --batch_size=128 --num_epochs=101 --val_every=5 --force_cpu --emb_dim=128 --learning_rate=0.001 --stemming

**NOTE: make sure epochs > val_every to get the plot for validation to have actual data on it, otherwise it's just a point.**

# Implementation Choices:  
My model architecture consists of an embedding layer, one LSTM hidden layer, and one fully connected output linear layer. Cross entropy is used as the loss function, and stochastic gradient descent is used for the optimizer.   
The embedding layer input is a 2D tensor of batch_size x len_cutoff, where len_cutoff is the number of words per instruction we learn, and outputs a 3D tensor with dimensions batch_size x len_cutoff x embed_dim.   
The LSTM takes the embeddings as inputs and outputs to the a 2D tensor of size embed_dim x embed_dim. I only used one hidden layer because it was enough to get a high accuracy. Two or more hidden layers would cause even more overfitting, considering the simplicity of this task.  
Finally, the fully connected output linear layer takes in the output from the LSTM and ouputs another 3D tensor of dimensions 1 x batch_size x num_actions+num_targets. After calling squeeze() to collapse the first axis (since we don't care too much about it), we're left with a 2D tensor of dimensions batch_size x num_actions+num_targets. I then split the tensor along its third axis into two tensors, one of size batch_size x num_actions and the other of size batch_size x num_targets and return these converted tensors as the final step of the forward pass. I chose to use one output layer as opposed to two independent output layers (one for action and one for target) because if I chose the later, then the model might not perform as well because it will learn the actions and targets independently. Since actions and targets are unlikely to be independent (for exmaple, the action "pick up" should be associated with targets such as "mug" and not "countertop"), the model performs better with this implementation. 

# Performance and hyperparameter tuning:  
This model learns performs very well on the training and validation data, usually reaching ~0.99 accuracy within the first epoch. However, this level of performance indicates that the model is overfitting on the data. This could be because a variety of reasons, such as the task being too simple, not enough data, or the training and validation sets having the same underlying distribution. The last is very likely to be a strong contributing factor, as there are many repeated examples in the training and validation data. This means that the model is unlikely to generalize well to new/unseen data.   
The performance with the different hyperparameters didn't vary too much, as long as they were kept within a certain range (not extremes, e.g. setting batch size to be 1000). The initial choice of hyperparameters was arbitrarily chosen by copying the ones in the skeleton code. 
batch_size: lowering batch size (to ~10) resulted in slightly lower accuracy for the first few epochs compared to raising it (~50)
num_epoch: 4-5 epochs is enough to guarantee consistent ~0.99 on the training and validation sets.
emb_dim: Lowering (to ~10) the embedding dimension (and since it's also used for the LSTM hidden layer, the hidden layer dimensions) didn't seem to have much effect at all. Increasing the embedding dimension (to ~1000) caused my laptop to freeze. A gentler increase (~200) ran, but didn't have much effect on performance.
learning_rate: finally, both lowering the learning rate (to ~0.0001) and increasing the learning rate (to ~1) didn't have much effect.

# Stemming:
Not much performance difference with stemming - perhaps due to the reasons specified above (simple task, not enough data, overlap/repeats with the training and validation sets). Additionally, there weren't that many words that were stemmed to begin with. Perhaps its because the instructions themselves are basic, with often repeated and atomic words, but looking at experiments/stemmedExamples.txt we see that there aren't that many stemmed words, and those that are stemmed aren't stemmed to a great extent (such as microwav), indicating that stemming isn't that useful in this dataset.  
Tuning the hyperparamters in a similar way as above led yielded similar results, i.e. no significant change in performance. 

# Other files
In the experiments directory there's trendsNoStemming.png, which contains the plots from running the code with default parameters (except epochs, which is set to 11) and no stemming. Additionally, there's rawTrainValidationScoreExamples.txt, which contain a few outputs (not complete) from the first few iterations.   
Furthermore, there's trendsStemming.png, which is the same as trendsNoStemming.png but with stemming. Finally, there's stemmedExamples.txt, which contains a few examples (not complete) of the stemmed outputs.  