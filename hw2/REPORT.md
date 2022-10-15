Leon Zha  
CSCI 499  
Jesse Thomason  
PA 2  

### **Requirements**
#### Note, this requires gensim 3.4.0, as using a later version of gensim (I had 4.2.0 before) causes dependency issues with gensim.utils.smart_open (attribute not found error)

####  **Assumptions and modification I've made:** 
##### I removed 2542.txt, 2554.txt, 4300.txt, and 5200.txt from the dataset because they were just html tags saying DO NOT SCRAPE etc. In encode_data() in data_utils.py I changed num_insts to be the number of instances of sentences rather than 2*# sentences. 
##### I made a lot of other changes to improve performance - those will be detailed in the metrics section

#### **Running this code**
##### Assuming you've got the same setup as found in the Github, the bare minimum command to run this code is:
##### `python train.py --data_dir=books/ --analogies_fn analogies_v3000_1309.json` 
##### Other flags may be added on to change the hyperparameters. 

#### **Model architecture**
##### The model has a very simple architecture that closely resembles the original skip-gram. It is simply composed of an embedded layer (initializing  with uniform distribution between -1 and 1) and a linear fully connected layer. For each batch, input words are passed into the model in a batch_size x 1 vector. This is passed into the embedding layer to return the embeddings for each word. Finally, these embeddings are passed into the linear fully connected layer. The output from the linear fully connected layer, along with the true context, is passed into BCE to calculate loss. An averaged IoU was used for calcuating accuracy. 

#### **Hyperparameter Tuning**
##### The important default parameters are as follows:
##### * vocab_size=3000
##### * batch_size=64
##### * num_epochs=31
##### * val_every=5
##### * save_every=5
##### * val_size=0.3
##### * emb_dim=128
##### * learning_rate=0.001
##### Unfortunately, my laptop kept crashing when using the default parameters due to lack of memory. Below, I list some avenues which I explored to address this issue. Please note that speed and memory usage is semi-subjective because they're based on self judgements (and looking at task manager), not actually timing each change. 

##### * Shrinking batch_size. When training the model, the two main structures that I concern myself with that are kept track of are the embedding layer, which has dimensions vocab_size x emb_dim (which is always >= word_embs in the forward() function, since word_embs is a subset of the emb_dim), and the logits that result after the linear fully connected layer, which has dimensions batch_size x vocab_size. Thus, if I shrink the batch size, I could save memory, albeit at the cost of running slower because it takes longer to get through an epoch. I ranged batch_size from 1 to 128, in powers of 2. 

##### * Shrinking vocab size. As mentioned above, vocabulary size plays a big role in how much memory is being used. I ranged vocab from 15k to the full 30k. Although it seemed to improve performance slightly, the model was still slow and took up a lot of memory, so I kept it at 30k for the default because I wasn't getting any better metrics. 

##### * I ranged num_epochs from 11 to 31. Similar to shrinking vocab size, when I shrank num_epochs, the model trained faster (since there were less epochs to run through), but overall was still slow and I wasn't meeting the metric requirements. 

##### * I ranged val_every from 1 to 10, with no success. More will be discussed when I talk about metrics below. 

##### * I ranged save_every the same way as val_every, hoping that saving less frequently would (minimally) speed up the training process because we don't stop to save as often. Once again, not a really noticable impact. 

##### * I increased the proportion of the data that goes into the validation set to try and save memory, ranging from 0.3 to 0.5. This had minimal impact, although I think validation was slower. 
##### * The emb_dim has an impact on how much memory is used for similar reasons as batch_size. I tried values ranging from 32 to 128 in powers of two. No noticable runtime or memory changes. 

#### Metrics
##### As mentioned above, my laptop kept crashing when trying to train my model. In addition to hyperparameter tuning, some other ways I tried to improve the memory usage and speed up training are:
##### * An assumption I made is that I only considered windows with no padding. That is, when creating (word, context) pairs, if the word was padding or the context contained padding, then I did not add that (word, context) pair to the training data.

#####  * I added a flag --save_encodings. By default, this is set to false, and so when it's not included in the list arguments passed in, my machine uses three .txt files, encoded_context.txt, encoded_words.txt, and index_to_vocab.txt, that contain the encoded data to speed up the training process. However, if you change the vocab size, then you'll have to call --save_encodings to re-save the encodings based off the new data. 

##### * Per Thomason's Slack reply "you only need in vitro accuracy for the val set; you don't need to calcuate a train set accuracy..."
##### (can be found here: https://uscviterbiclass.slack.com/archives/C03QP2U3BFV/p1665681457209679?thread_ts=1665671601.099869&cid=C03QP2U3BFV)
##### at one point I only calculated accuracy every val_every epochs, because calculating the IoU accuracy was slow. Calculating IoU accuracy was slow because of I had to get the indices of the words that actually appear around the current word from the actual_context, then sort the probabilities in the predicted context and take the top probabilities there. I later reverted this change because there weren't any noticable speed ups, I wanted to keep track of the accuracy, and the directions did say to report training accuracy as well as validation accuracy. 

##### * To further optimize IoU_accuracy's runtime, I used two tricks: I passed in the indices of the actual context directly (as opposed to converting them back from their multihot encoded format) and I used numpy's argpartition() method to efficiently partition the list of predicted probabilities. 

##### * To save memory at the expense of runtime, rather than reading in all data and creating the dataloaders at the very beginning, I only create the dataloaders for training/validation when it's their turn. However, since I have to read in the data for this, it does cost some compute time. 

##### * Two other resources I explored are USC's High Performance Computing (HPC) Discovery Cluster and Google Colab. The HPC Cluster was a dead end because from what I understand, in order to submit job batches, we need a 'project' folder, which is only assigned to PIs. Google Colab was also a dead end because while I was able to take advantage of its GPU, I once again ran out of RAM (only 12GB for free users). 

##### * Finally, I cut down the number of training instances because even if it was slow, I just wanted *some* results rather than having my laptop crash on me partway through training. In my code, I hard code the amount of training instances to consider to be 1.5 million. This move sacrifices accuracy, but I was able to get the code runing on my laptop with the following hyperparameters: 
##### `python train_xtreme_memory_saving.py --data_dir=books/ --analogies_fn analogies_v3000_1309.json --vocab_size=3000 --batch_size=64 --num_epochs=20 --val_every=5 --save_every=5 --val_size=0.3 --emb_dim=64 --window_size=4`

#### **Performance**
##### All metrics are harvested from the above instance, which took ~14 hours to run. Results can be found in the folder batch_64_epochs_20_val_0.3_emb_dim_64_results. 
##### In vitro: Loss steadily decreased over training. Since loss actually exhibited change, I created a graph and put it into the batch_64_epochs_20_val_0.3_emb_dim_64_results folder. Accuracy was still 0, or close to 0 (I got a few numbers that looked like 9.523809523809524e-08). Validation loss and accuracy had a similar trend. I also created a graph for validation loss. 
##### In vivo: Total performance across all 1309 analogies: 0.0000 (Exact); 0.0020 (MRR); 505 (MR). Analogy performance across 969 "sem" relation types: 0.0000 (Exact); 0.0019 (MRR); 533 (MR). Analogy performance across 340 "syn" relation types: 0.0000 (Exact); 0.0023 (MRR); 440 (MR). Since over all 1309 analogies I got 0 correct, I'm below the 0.0003 threshold. 


#### **Analysis of given code**
##### The in vitro task is the task we train our model on, that is, the validation set. We're trying to predict what the context is when given a word. Since I chose to implement a skip-gram model, I'm using Intersect over Union (IoU) to as a metric to ascertain the accuracy of the model. IoU is a useful metric because the model can't get away with simply predicting a lot of things because it uses the intersection of predicted and actual context words. The in vivo task is the downstream application that the model wasn't explicitly trained to do, in our case the analogy evaluation. We're trying to see if similar words have vectors that are close in vector space. The downstream application has three metrics: exact, MRR, and MR. Exact means an exact match, that is, the closest predicted vector is exactly what the analogous vector should be. As for MR and MRR, MR is just 1/MRR, so I'll focus on MRR. MRR is mean reciprocal rank. MRR gives a general measure of how close the correct vector is among the surrounding vectors where we end up after doing the analogy process. 