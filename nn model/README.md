# Model Architecture
The architecture of the MLP model is shown in the figure:
![image](https://github.com/karrich/KDD-CUP-2023-solution/assets/57396778/0621c3b0-2654-4e62-a5e1-4f73158b021d)
We use the ID of the product to obtain the 128-dimensional embedding of the product. 

For the input session, we perform the operation shown in the figure to obtain a 4*128-dimensional embedding,
and then input MLP to get a 128-dimensional session embedding.

For Target Embedding, we use its ID Embedding directly.

Than,we optimize the cosine similarity between session embedding and target embedding.
# Detail
## Data augmentation
![image](https://github.com/karrich/KDD-CUP-2023-solution/assets/57396778/5c928907-e5a5-43d1-8b94-ad0d9fd55e10)
For a complete training session, we will split it into multiple training sessions.
And we increased the proportion of complete sessions in the dataset.
## Embedding initialization
We initialize the embedding space of the product ID through a 128-dimensional embedding of the title information of the product.
This effectively boosts the results.
## Negative sampling
The method of negative sampling is 'Negative in-batch sampling'.
And we treat session embedding and target embedding fairly.
This means that for a session embedding, its negative sample includes other session embeddings in the batch.
## test data
When building the training data, we used the test dataset.
And we experimentally found that this data is very important.
