Leon Zha  
CSCI 499  
Jesse Thomason  
PA 2  

### **Requirements**
#### Note, this requires gensim 3.4.0, as using a later version of gensim (I had 4.2.0 before) causes dependency issues with gensim.utils.smart_open (attribute not found error)

####  **Assumptions and modification I've made:** 
##### I removed 2542.txt, 2554.txt, 4300.txt, and 5200.txt from the dataset because they were just html tags saying DO NOT SCRAPE etc. In encode_data() in data_utils.py I changed num_insts to be the number of instances of sentences. To save memory during data processing, I only considered windows with no padding. That is, when creating (word, context) pairs, if the word was padding or the context contained padding, then I did not add that (word, context) pair to the training data.   
##### I also added several flags, one of which, --save_encodings, I want to bring attention to. By default, this is set to false, and so when it's not included in the list arguments passed in, my machine uses three .txt files, encoded_context.txt, encoded_words.txt, and index_to_vocab.txt, that contain the encoded data to speed up the training process. However, if you change the vocab size, then you'll have to call --save_encodings to re-save the encodings based off the new data. 

#### **Running this code**
##### Assuming you've got the same setup as found in the Github, the bare minimum command to run this code is:
##### `train.py --data_dir=books/ --analogies_fn analogies_v3000_1309.json` 
##### Other flags may be added on to change the hyperparameters. 

#### **Hyperparameter choice**
##### Unfortunately, my laptop wasn't strong enough to run the above command with the defaults as is. I used the `--val_size` flag to set the validation set to be 0.4 of the total data, as opposed to the default 0.3. As for the other hyperparamters, most were chosen arbitrarily with an upper bound to account for memory and training time. For example, consider the number of features of a vector. In the original skip-gram model, they learned 300 features for every word. Since I don't have a machine as powerful, nor as much time, I chose to learn less features. I settled on 128 because it's still a decent amount, but also less than 300. Another example is the batch_size. The default batch_size is 32, which while fiarly small, must be small to account for the compute power of my machine. Vocab_size and num_epochs were left default from the skeleton code. Learning rate was chosen arbitrarily based on what I've seen online. 

#### **Model architecture**
##### The model has a very simple architecture that closely resembles the original skip-gram. It is simply composed of an embedded layer (initializing  with uniform distribution between -1 and 1) and a linear fully connected layer. For each batch, input words are passed into the model in a batch_size x 1 vector. This is passed into the embedding layer to return the embeddings for each word. Finally, these embeddings are passed into the linear fully connected layer. The output from the linear fully connected layer, along with the true context, is passed into BCE to calculate loss. IoU was used for calcuating accuracy. 

#### **Performance**
##### In vitro:

#### In vivo:


#### **Analysis of given code**
##### The in vitro task is the task we train our model on, that is, the validation set. We're trying to predict what the context is when given a word. Since I chose to implement a skip-gram model, I'm using Intersect over Union (IoU) to as a metric to ascertain the accuracy of the model. 

The in vivo task is the downstream application that the model wasn't explicitly trained to do, in our case the analogy evaluation. We're trying to see if similar words have vectors that are close in vector space. 