Leon Zha  
CSCI 499  
Jesse Thomason  
PA 2  

#### **Running the code**
##### Assuming you've got the same setup as found in the Github, the bare minimum command to run this code is:
##### `python ./train.py --in_data_fn="./lang_to_sem_data.json" --model_output_dir="./results"` 

##### Other flags may be added on to change the hyperparameters. 
##### If you wish to see accuracy and loss of a new run, you should delete the .txt files for them in the ./results folder. 

####  **Dataset:** 
##### I used the updated dataset from after Tuesday, 11/15, so the maximum number of examples in an episode isn't bugged and really big


#### **Model architecture**
##### Base Seq2Seq
###### Encoder: The encoder is super simple, consisting of only an embedding layer and an lstm. A sequence is passed into the embedding layer to get the embeddings for each word, then those embeddings are passed through the LSTM. We keep the hidden and cell states returned as outputs from the LSTM for use in the decoder.

###### Decoder: The decoder consists of two embedding layers, an LSTM, and two fully connected linear layers. There are two embedding layers and two fully connected layers because one is for targets, and one for actions. The inputs into the embedding layers are batch_size x seq_len. After getting the embeddings for actions and targets, I concatenate them together along the first dimension, resulting in a batch_size x 2 x embedding_dim tensor. This is passed into the LSTM, along with the previous hidden and cell state to get the output and the new hidden and cell states. Finally, I extract the predicted_action and predicted_target layers from the output and return them.

###### EncoderDecoder: The EncoderDecoder is a wrapper that goes around the encoder and decoder models. First, instructions are passed into the encoder. Then, I loop through each pair of (action, target) labels. If it's the first time we're using the decoder, then what we're passing in as the true labels are the "<bos>" tokens used to indicate it's the beginning of a sequence. If it's not the first input and we're training, then I pass in the true labels because we're teacher forcing. The instructions implied we would do teacher forcing for every training instance, but I only used it based on a probability, which is one of the input flags. 



#### **Hyperparameter Tuning**
##### I couldn't run the model in a reasonable amount of time so I cut down on the vocabulary size, number of training and validation instances we condider, and number of epochs. Thus, while the command at the very beginning represented the bare minimum flags required to run this code, the actual command I used was: `python ./train.py --in_data_fn="./lang_to_sem_data.json" --model_output_dir="./results" --num_epochs=51 --train_cutoff=500 --val_cutoff=500 --vocab_size=100`. All the printed metrics are using these hyperparameters. Other hyperparameters you can modeify using flags if you wished are:

##### batch_size: by default 64. Chosen kind of arbitrarily, but works well based on what I've seen in other models. I changed it when playing around with getting the model to train faster but in the end it wasn't enough so I changed the other hyperparameters instead. Note that since the embedding layers depend on batch_size, if the number of inputs from  the dataloader aren't evenly divisible by the batch_size (i.e. if len(data_loader)%batch_size != 0) then there's an error because of mismatch in dimensions. To prevent this, while training, if the lenth of the inputs is not equal to the batch_size then I don't train on it. This should only ever exclude the last n examples in the dataloader, where n is in [1, batch_size-1]. 
##### val_every: by default 5. Also chosen arbitrarily. DIdn't have much of an effect on accuracy when I played with it so I kept it as is. 
##### emb_dim: by default 128. Much like batch_size, I chose it kind of arbitrarily based on what's worked well in other models. I changed it when playing around with getting the model to train faster but in the end it wasn't enough so I changed the other hyperparameters instead.
##### learning_rate: chosen arbitrarily based on what's worked well before. Didn't touch it at all.
##### train_cutoff: by default 1000000000. Represents an upper limit for how many training instances to cover. If the number of training examples is less than train_cutoff, we'll use the number of training examples instead. 
##### val_cutoff: Same as train_cutoff, but for the validation instances.
##### vocab_size: default 1000. the number of words in our vocabulary. 
##### teacher_forcing_prob: default 0.9. The probability we use teacher forcing during training time.



#### **Performance**
##### Training and validation metrics can be found in the `results` folder. Note that if you re-run the code, delete the contents of the `results` folder because metrics are appended to each file, so you'd end up getting old metrics + new metrics if you didn't delete. 
##### training loss can be found in training_loss.txt, validation loss and accuracy in val_metrics.txt (in that order, with a space separating the two. So each row represents a validation run, the 0th indexed column is the loss, and the 1st indexed column represents the accuracy). To get a summmary of these, we can look at train_val_metrics_plot.png. Training loss decreases pretty steadily, which is expected. Validation loss decreases, then increases, which is surprising considering validation accuracy goes up. To be fair though, since the amount of data being fed in is so small, this could be random noise (looking at the x-axis, the absolute values barely change).




#### References used during this homework: 
##### https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

##### https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

##### https://www.guru99.com/seq2seq-model.html