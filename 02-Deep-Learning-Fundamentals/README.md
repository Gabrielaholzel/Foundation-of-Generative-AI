# Deep Learning Fundamentals

**Table of Contents**
- [Perceptron](#perceptron)
  * [Key Concepts](#key-concepts)
- [The Multi-Layer Perceptron](#the-multi-layer-perceptron)
  * [Key Concepts](#key-concepts-1)
- [Training Deep Neural Networks](#training-deep-neural-networks)
- [PyTorch](#pytorch)
  * [PyTorch Tensors](#pytorch-tensors)
    + [Key Concepts](#key-concepts-2)
  * [PyTorch Neural Networks](#pytorch-neural-networks)
  * [PyTorch Loss Functions](#pytorch-loss-functions)
    + [Code Examples](#code-examples)
      - [Cross-Entropy Loss](#cross-entropy-loss)
      - [Mean Squared Error Loss](#mean-squared-error-loss)
  * [PyTorch Optimizers](#pytorch-optimizers)
    + [Code Examples](#code-examples-1)
      - [Stochastic Gradient Descent](#stochastic-gradient-descent)
      - [Adam](#adam)
  * [PyTorch Datasets and Data Loaders](#pytorch-datasets-and-data-loaders)
    + [Code Examples](#code-examples-2)
      - [Datasets](#datasets)
      - [Data Loaders](#data-loaders)
  * [PyTorch Training Loops](#pytorch-training-loops)
    + [Code Examples](#code-examples-3)
      - [Create a Number Sum Dataset](#create-a-number-sum-dataset)
      - [Inspect the Dataset](#inspect-the-dataset)
      - [Define a Simple Model](#define-a-simple-model)
      - [Instantiate Components Needed for Training](#instantiate-components-needed-for-training)
      - [Create a Training Loop](#create-a-training-loop)
      - [Try the Model Out](#try-the-model-out)
- [Hugging Face](#hugging-face)
  * [Tokenizers](#tokenizers)
    + [Code Example](#code-example)
  * [Face Models](#face-models)
    + [Code Example](#code-example-1)
  * [Datasets](#datasets-1)
    + [Code Example](#code-example-2)
  * [Trainers](#trainers)
    + [Code Example](#code-example-3)
- [Pre-Trained Models and Transfer Learning](#pre-trained-models-and-transfer-learning)


## Perceptron
A perceptron is an essential component in the world of AI, acting as a binary classifier capable of deciding whether data, like an image, belongs to one class or another. It works by adjusting its weighted inputs—think of these like dials fine-tuning a radio signal—until it becomes better at predicting the right class for the data. This process is known as learning, and it shows us that even complex tasks start with small, simple steps.

![Diagram showing pixels and weights being added together, passed through an activation function, then returning a cat or dog classification](https://video.udacity-data.com/topher/2023/December/658b1687_genai-nd-c1-l2-perceptron/genai-nd-c1-l2-perceptron.jpeg)
1. **How It Works**:
    -   The perceptron receives multiple inputs and assigns a weight to each input.
    -   It multiplies each input by its corresponding weight and sums the results.
    -   This sum is then passed through an activation function (originally a step function) to produce an output of either 0 or 1, indicating the classification.
2. **Learning Process**:
	The perceptron learns by adjusting its weights based on the errors it makes during predictions. This process involves comparing the predicted output with the actual class and tweaking the weights to improve accuracy over time.
3. **Limitations**: A single perceptron can only solve linearly separable problems. However, when combined in layers, multiple perceptrons can tackle more complex tasks.
4. **Activation Functions**: A mathematical equation that decides whether the perceptron's calculated sum from the inputs is enough to trigger a positive or negative output. Modern neural networks often use more complex functions like ReLU (Rectified Linear Unit) to allow for a broader range of outputs.


### Key Concepts
* **Binary Classifier:**  A type of system that categorizes data into one of two groups. Picture a light switch that can be flipped to either on or off.

* **Vector of Numbers:**  A sequence of numbers arranged in order, which together represent one piece of data.

* **Activation Function:**  A mathematical equation that decides whether the perceptron's calculated sum from the inputs is enough to trigger a positive or negative output.


## The Multi-Layer Perceptron
The multi-layer perceptron is a powerful tool in the world of machine learning, capable of making smart decisions by mimicking the way our brain's neurons work. This amazing system can learn from its experiences, growing smarter over time as it processes information through layers, and eventually, it can predict answers with astonishing accuracy!

**Structure**:
* **Input Layer**: The first layer that receives raw data.
* **Hidden Layers**: One or more layers between the input and output layers that perform complex transformations on the data.
* **Output Layer**: The final layer that produces predictions or decisions based on the processed data.

**Learning Mechanism**
The MLP learns from experience by adjusting the weights associated with connections between neurons during training. This adjustment aims to minimize errors in predictions.

**Functionality**
Each neuron in the hidden layers is connected to every input, and these connections have weights that are modified during training. The output layer's neurons correspond to different classes, with the neuron producing the highest value indicating the predicted class.

**Applications**
MLPs can handle complex, high-dimensional datasets, making them suitable for tasks such as image recognition and classification.


### Key Concepts
* **Multi-Layer Perceptron (MLP):**  A type of artificial neural network that has multiple layers of nodes, each layer learning to recognize increasingly complex features of the input data.
* **Input Layer:**  The first layer in an MLP where the raw data is initially received.
* **Output Layer:**  The last layer in an MLP that produces the final result or prediction of the network.
* **Hidden Layers:**  Layers between the input and output that perform complex data transformations.


## Training Deep Neural Networks
**Labelled Dataset**
A labelled dataset consists of input data paired with corresponding output labels. This pairing is crucial for the model to learn and make accurate predictions.

**Gradient Descent**
This is an optimization algorithm used to minimize the cost function of a neural network. It works by adjusting the model's parameters (weights) in the direction that reduces the error, similar to finding the lowest point in a valley.

**Cost Function**
The cost function measures how well the neural network's predictions match the actual labels. The goal is to minimize this function during training.

**Learning Rate**
This hyperparameter determines the size of the steps taken during the optimization process. A learning rate that is too high may cause the model to overshoot the optimal solution, while a rate that is too low can lead to slow convergence.

**Backpropagation**
This method involves calculating the gradient of the cost function with respect to each weight by propagating the error backward through the network. It allows the model to update its weights based on the error from the previous iteration.

**Testing the Model**
After training, it's essential to test the model on a separate dataset (holdout dataset) to evaluate its performance and generalization capabilities.

## PyTorch
PyTorch is a dynamic and powerful tool for building and training machine learning models. It simplifies the process with its fundamental building blocks like tensors and neural networks and offers effective ways to define objectives and improve models using loss functions and optimizers. By leveraging PyTorch, anyone can gain the skills to work with large amounts of data and develop cutting-edge AI applications.

### PyTorch Tensors
PyTorch tensors are crucial tools in the world of programming and data science, which work somewhat like building blocks helping to shape and manage data effortlessly. These tensors allow us to deal with data in multiple dimensions, which is especially handy when working with things like images or more complex structures. 

**Definition of Tensors**
Tensors are multi-dimensional arrays that can hold data in various dimensions, similar to vectors and matrices but with the capability to have more than two dimensions.

**Types of Data**
Tensors can store different types of data, including scalar values, vectors, and higher-dimensional entities. They are essential for representing data in machine learning tasks.

**Creation of Tensors**
You can create tensors in PyTorch, including initializing them with random values or specific data types.
```python
import torch 

# Create a 3-dimensional tensor 
images = torch. rand((4, 28, 28)) 

# Get the second image 
second_image = images (1]
```

**Operations on Tensors**
Tensors support a wide range of operations, such as addition, multiplication, and reshaping, which are crucial for performing computations in deep learning models.

```python
a = torch.tensor([[1,  1],  [1,  0]])  
print(a)  
# tensor([[1, 1],  
#	      [1, 0]])  

print(torch.matrix_power(a,  2))  
# tensor([[2, 1],  
#	 	  [1, 1]])  

print(torch.matrix_power(a,  3))  
# tensor([[3, 2],  
# 	 	  [2, 1]])  

print(torch.matrix_power(a,  4))  
# tensor([[5, 3],  
#		  [3, 2]])
```

**Visualization**
You can display an image from a tensor using the Matplotlib library.
```python
import matplotlib.pyplot as plt 

# Display the image 
plt. imshow(second_image, cmap='gray') 
plt.axis('off') # disable axes 
plt. show()
```

#### Key Concepts
**Tensors**: Generalized versions of vectors and matrices that can have any number of dimensions (i.e. multi-dimensional arrays). They hold data for processing with operations like addition or multiplication.

**Matrix operations:**  Calculations involving matrices, which are two-dimensional arrays, like adding two matrices together or multiplying them.

**Scalar values:**  Single numbers or quantities that only have magnitude, not direction (for example, the number 7 or 3.14).

**Linear algebra:**  An area of mathematics focusing on vector spaces and operations that can be performed on vectors and matrices.


### PyTorch Neural Networks
Neural networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. Each neuron processes input data and passes its output to the next layer.

**Structure**:
* **Input Layer**: The first layer that receives input data.
* **Hidden Layers**: Intermediate layers where computations occur. There can be multiple hidden layers in a deep neural network.
* **Output Layer**: The final layer that produces the output or prediction based on the processed data.

**Activation Functions**
These functions determine whether a neuron should be activated (i.e., produce an output) based on the input it receives. Common activation functions include ReLU (Rectified Linear Unit) and sigmoid functions.

**Training Process**
* **Forward Propagation**: Input data is passed through the network, layer by layer, to generate an output.
* **Loss Function**: A metric that measures the difference between the predicted output and the actual target value. The goal is to minimize this loss during training.
* **Backpropagation**: A method used to update the weights of the network based on the error calculated from the loss function. This involves calculating gradients and adjusting weights to improve accuracy.

**Applications**
Neural networks are widely used in various applications, including image recognition, natural language processing, and game playing, due to their ability to learn complex patterns from data.

**Code Example**
```python
import torch.nn as nn 

class  MLP(nn.Module):
	def  __init__(self, input_size):
		super(MLP, self).__init__()
		self.hidden_layer = nn.Linear(input_size,  64)
		self.output_layer = nn.Linear(64,  2) 
		self.activation = nn.ReLU()
	
	def  forward(self, x):
		x = self.activation(self.hidden_layer(x))
		return self.output_layer(x)  model = 

MLP(input_size=10)
print(model)

# MLP(  
#	 (hidden_layer): 
#	Linear(in_features=10, out_features=64, bias=True)  
#	 (output_layer):
#	Linear(in_features=64, out_features=2, bias=True)  
# 	(activation):
#	ReLU()
# )

model.forward(torch.rand(10))
# tensor([0.2294, 0.2650], grad_fn=<AddBackward0>)
```

### PyTorch Loss Functions
PyTorch loss functions are essential tools that help in improving the accuracy of a model by measuring errors. These functions come in different forms to tackle various problems, like deciding between categories (classification) or predicting values (regression).

**Purpose of Loss Functions**
Loss functions quantify the difference between the predicted outputs of a model and the actual target values. They are essential for measuring how well the model is performing.

**Common Loss Functions**
-   **Cross-Entropy Loss**: Used primarily for classification tasks, it measures how well the predicted probabilities align with the actual classes. It's particularly effective when classes are mutually exclusive.
-   **Mean Squared Error (MSE)**: Commonly used for regression tasks, it calculates the average of the squared differences between predicted and actual values.

**Minimizing Loss**
The goal during training is to minimize the loss value, which indicates that the model's predictions are becoming more accurate.

**Implementation in PyTorch**
We can also see how to implement these loss functions using PyTorch's `torch.nn` module.

**Importance of Choosing the Right Loss Function**
Selecting an appropriate loss function is crucial as it impacts the model's ability to generalize and perform well on unseen data.

#### Code Examples

##### Cross-Entropy Loss

```python
import torch 
import torch.nn as nn 

loss_function = nn.CrossEntropyLoss()  

# Our dataset contains a single image of a dog, where  
# cat = 0 and dog = 1 (corresponding to index 0 and 1)  
target_tensor = torch.tensor([1])  
target_tensor 
# tensor([1])
```

Prediction: Most likely a dog (index 1 is higher)
```python
# Note that the values do not need to sum to 1  
predicted_tensor = torch.tensor([[2.0,  5.0]])  
loss_value = loss_function(predicted_tensor, target_tensor)  
loss_value 
# tensor(0.0181)
```

Prediction: Slightly more likely a cat (index 0 is higher)
```python
predicted_tensor = torch.tensor([[1.5,  1.1]])  
loss_value = loss_function(predicted_tensor, target_tensor)
loss_value 
# tensor(0.9130)
```

##### Mean Squared Error Loss
```python
# Define the loss function  
loss_function = nn.MSELoss()  

# Define the predicted and actual values as tensors  
predicted_tensor = torch.tensor([320000.0])  
actual_tensor = torch.tensor([300000.0])  

# Compute the MSE loss  
loss_value = loss_function(predicted_tensor, actual_tensor)
print(loss_value.item())  
# Loss value: 20000 * 20000 / 1 = ...  
# 400000000.0
```


### PyTorch Optimizers
PyTorch optimizers are important tools that help improve how a neural network learns from data by adjusting the model's parameters. By using these optimizers, like stochastic gradient descent (SGD) with momentum or Adam, we can quickly get started learning!

**Reinforcement Learning**
RL is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions.

**Components of RL**
-   **Agent**: The learner or decision-maker.
-   **Environment**: The context within which the agent operates.
-   **Actions**: Choices made by the agent that affect the environment.
-   **Rewards**: Feedback received from the environment, guiding the agent's learning.

**Learning Process**
-   The agent explores different actions to discover which ones yield the highest rewards.
-   Over time, the agent learns to optimize its strategy to maximize cumulative rewards through trial and error.

**Exploration vs. Exploitation**
The agent must balance exploring new actions to find better rewards (exploration) with leveraging known actions that yield high rewards (exploitation).

**Applications of RL**
RL is used in various fields, including robotics, game playing (e.g., AlphaGo), and autonomous vehicles, where decision-making in dynamic environments is crucial.

#### Code Examples

Assuming  `model`  is your defined neural network.
`lr=0.01`  sets the learning rate to 0.01 for either optimizer.

```python
import torch.optim as optim
```

#####  Stochastic Gradient Descent
```python
# momentum=0.9 smoothes out updates and can help training 
optimizer = optim.SGD(
					model.parameters(), 
					lr=0.01, 
					momentum=0.9
					)
```
##### Adam
```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```


### PyTorch Datasets and Data Loaders

**Data Augmentation**
Data augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of existing data. This helps improve the model's ability to generalize to new, unseen data.

**Importance**
Augmenting data is crucial when the available dataset is small, as it can help prevent overfitting, where the model learns to perform well on the training data but fails to generalize to new data.

**Common Techniques**
-   **Image Augmentation**: Techniques include rotation, flipping, scaling, cropping, and colour adjustments. These transformations create variations of images while preserving their labels.
-   **Text Augmentation**: Methods may involve synonym replacement, random insertion, or back-translation to generate diverse text samples.

**Benefits**
By increasing the diversity of the training data, data augmentation can lead to improved model performance, robustness, and accuracy in real-world applications.

#### Code Examples

##### Datasets
```python
from torch.utils.data import Dataset 

# Create a toy dataset  
class  NumberProductDataset(Dataset):
	def  __init__(self, data_range=(1,  10)):
		self.numbers = list(range(data_range[0],data_range[1]))
	
	def  __getitem__(self, index):
		number1 = self.numbers[index]
		number2 = self.numbers[index]  +  1
		return  (number1, number2), number1 * number2 
	
	def  __len__(self):
	return  len(self.numbers)  

# Instantiate the dataset  
dataset = NumberProductDataset(
	data_range=(0,  11)
)  
# Access a data sample  
data_sample = dataset[3]
print(data_sample)
# ((3, 4), 12)
```

##### Data Loaders
```python
from torch.utils.data import DataLoader 

# Instantiate the dataset  
dataset = NumberProductDataset(data_range=(0,  5))  

# Create a DataLoader instance  
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)  

# Iterating over batches  
for  (num_pairs, products)  in dataloader:
	print(num_pairs, products)
# [tensor([4, 3, 1]), tensor([5, 4, 2])] tensor([20, 12, 2]) 
# [tensor([2, 0]), tensor([3, 1])] tensor([6, 0])
```

### PyTorch Training Loops
A PyTorch training loop is an essential part of building a neural network model, which helps us teach the computer how to make predictions or decisions based on data. By using this loop, we gradually improve our model's accuracy through a process of learning from its mistakes and adjusting.

The training loop is a cycle that the model goes through multiple times to learn from the data. It involves several key components, including the dataset, model, loss function, and optimizer.

**Epochs and Batches**
-   **Epoch**: One complete pass through the entire training dataset.
-   **Batch**: A subset of the training data used to update the model's weights during each iteration.

**Steps in the Training Loop**
1. Initialize the total loss to zero at the beginning of each epoch.
2. Iterate over batches of data, making predictions with the model.
3. Calculate the loss by comparing the model's predictions to the actual target values.
4. Update the total loss and adjust the model's weights using the optimizer based on the calculated loss.

**Monitoring Progress**
After each epoch, the training loop can print the loss to monitor the model's performance over time, ideally showing a decrease in loss as the model learns.

#### Code Examples

##### Create a Number Sum Dataset

This dataset has two features—a pair of numbers—and a target value—the sum of those two numbers.

Note that this is  _not_  actually a good use of deep learning. At the end of our training loop, the model still doesn't know how to add 3 + 7! The idea here is to use a simple example so it's easy to evaluate the model's performance.
```python
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 

class  NumberSumDataset(Dataset):
	def  __init__(self, data_range=(1,  10)):
		self.numbers =  list(range(data_range[0], data_range[1]))
	
	def  __getitem__(self, index):
		number1 =  float(self.numbers[index //  len(self.numbers)])
		number2 =  float(self.numbers[index %  len(self.numbers)])
		return torch.tensor([number1, number2]), torch.tensor([number1 + number2])
	
	def  __len__(self):
		return  len(self.numbers)  **  2
```

##### Inspect the Dataset
```python
dataset = NumberSumDataset(data_range=(1,  100))

for i in  range(5):
	print(dataset[i])

# (tensor([1., 1.]), tensor([2.]))
# (tensor([1., 2.]), tensor([3.]))  
# (tensor([1., 3.]), tensor([4.]))  
# (tensor([1., 4.]), tensor([5.]))  
# (tensor([1., 5.]), tensor([6.]))
```

##### Define a Simple Model
```python
class  MLP(nn.Module):
	
	def  __init__(self, input_size):
		super(MLP, self).__init__()
		self.hidden_layer = nn.Linear(input_size,  128)
		self.output_layer = nn.Linear(128,  1)
		self.activation = nn.ReLU()
	
	def  forward(self, x):
		x = self.activation(self.hidden_layer(x))
		return self.output_layer(x)
```

##### Instantiate Components Needed for Training
```python
dataset = NumberSumDataset(data_range=(0,  100))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = MLP(input_size=2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

##### Create a Training Loop
```python
for epoch in  range(10):
	total_loss =  0.0
	for number_pairs, sums in dataloader:  # Iterate over the batches
		predictions = model(number_pairs)  # Compute the model output
		loss = loss_function(predictions, sums)  # Compute the loss
		loss.backward()  # Perform backpropagation  
		optimizer.step()  # Update the parameters  
		optimizer.zero_grad()  # Zero the gradients  
	
		total_loss += loss.item()  # Add the loss for all batches  
	
	# Print the loss for this epoch  
	print("Epoch {}: Sum of Batch Losses = {:.5f}".format(epoch, total_loss))  

# Epoch 0: Sum of Batch Losses = 118.82360  
# Epoch 1: Sum of Batch Losses = 39.75702  
# Epoch 2: Sum of Batch Losses = 2.16352  
# Epoch 3: Sum of Batch Losses = 0.25178  
# Epoch 4: Sum of Batch Losses = 0.22843  
# Epoch 5: Sum of Batch Losses = 0.19182  
# Epoch 6: Sum of Batch Losses = 0.15507  
# Epoch 7: Sum of Batch Losses = 0.07789  
# Epoch 8: Sum of Batch Losses = 0.06329  
# Epoch 9: Sum of Batch Losses = 0.04936
```

##### Try the Model Out
```python
# Test the model on 3 + 7  
model(torch.tensor([3.0,  7.0]))  

# tensor([10.1067], grad_fn=<AddBackward0>)
```

## Hugging Face
Hugging Face is a company making waves in the technology world with its amazing tools for understanding and using human language in computers. Hugging Face offers everything from tokenizers, which help computers make sense of text, to a huge variety of ready-to-go language models, and even a treasure trove of data suited for language tasks.

**Tokenizers:**  
These work like a translator, converting the words we use into smaller parts and creating a secret code that computers can understand and work with.

**Models:** 
These are like the brain for computers, allowing them to learn and make decisions based on information they've been fed.

**Datasets:**  
Think of datasets as textbooks for computer models. They are collections of information that models study to learn and improve.

**Trainers:**  
Trainers are the coaches for computer models. They help these models get better at their tasks by practicing and providing guidance. Hugging Face Trainers implement the PyTorch training loop for you, so you can focus instead on other aspects of working on the model.

### Tokenizers
HuggingFace tokenizers help us break down text into smaller, manageable pieces called tokens. These tokenizers are easy to use and also remarkably fast due to their use of the Rust programming language.

**Tokenization**
It's like cutting a sentence into individual pieces, such as words or characters, to make it easier to analyse.

**Tokens** 
These are the pieces you get after cutting up text during tokenization, kind of like individual Lego blocks that can be words, parts of words, or even single letters. These tokens are converted to numerical values for models to understand.

**Pre-trained Model**  
This is a ready-made model that has been previously taught with a lot of data.

**Uncased**  
This means that the model treats uppercase and lowercase letters as the same.


#### Code Example
```python
from transformers import BertTokenizer 

# Initialize the tokenizer  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  

# See how many tokens are in the vocabulary  
tokenizer.vocab_size 
# 30522
```

```python
# Test the model on 3 + 7  
# Tokenize the sentence  
tokens = tokenizer.tokenize("I heart Generative AI")  

# Print the tokens  
print(tokens)  
# ['i', 'heart', 'genera', '##tive', 'ai']  

# Show the token ids assigned to each token  
print(tokenizer.convert_tokens_to_ids(tokens))  
# [1045, 2540, 11416, 6024, 9932]
```

### Face Models
Hugging Face models provide a quick way to get started using models trained by the community. With only a few lines of code, you can load a pre-trained model and start using it on tasks such as sentiment analysis.

#### Code Example
```python
from transformers import BertForSequenceClassification, BertTokenizer 

# Load a pre-trained sentiment analysis model  
model_name =  "textattack/bert-base-uncased-imdb"  
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  

# Tokenize the input sequence  
tokenizer = BertTokenizer.from_pretrained(model_name) 
inputs = tokenizer("I love Generative AI", return_tensors="pt")  

# Make prediction  
with torch.no_grad():
	outputs = model(**inputs).logits 
	probabilities = torch.nn.functional.softmax(outputs, dim=1)  
	predicted_class = torch.argmax(probabilities)  

# Display sentiment result  
if predicted_class ==  1:
	print(f"Sentiment: Positive ({probabilities[0][1]  *  100:.2f}%)")
else:
	print(f"Sentiment: Negative ({probabilities[0][0]  *  100:.2f}%)")  
# Sentiment: Positive (88.68%)
```

### Datasets
HuggingFace Datasets library is a powerful tool for managing a variety of data types, like text and images, efficiently and easily. This resource is incredibly fast and doesn't use a lot of computer memory, making it great for handling big projects without any hassle.

**IMDb dataset**  
A dataset of movie reviews that can be used to train a machine learning model to understand human sentiments.

**Apache Arrow** 
A software framework that allows for fast data processing.


#### Code Example
```python
from datasets import load_dataset
from IPython.display import HTML, display

# Load the IMDB dataset, which contains movie reviews
# and sentiment labels (positive or negative)
dataset = load_dataset("imdb")

# Fetch a revie from the training set
review_number = 42
sample_review = dataset["train"][review_number]

display(HTML(sample_review["text"][:450] + "..."))
# WARNING: This review contains SPOILERS. Do not read if you don't want some points revealed to you before you watch the
# film.
# 
# With a cast like this, you wonder whether or not the actors and actresses knew exactly what they were getting into. Did they
# see the script and say, `Hey, Close Encounters of the Third Kind was such a hit that this one can't fail.' Unfortunately, it does.
# Did they even think to check on the director's credentials...

if sample_review["label"] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
# Sentiment: Negative
```


### Trainers
Hugging Face trainers offer a simplified approach to training generative AI models, making it easier to set up and run complex machine learning tasks. This tool wraps up the hard parts, like handling data and carrying out the training process, allowing us to focus on the big picture and achieve better outcomes with our AI endeavours.

**Truncating**
This refers to shortening longer pieces of text to fit a certain size limit.

**Padding**
Adding extra data to shorter texts to reach a uniform length for processing.

**Batches** 
Batches are small, evenly divided parts of data that the AI looks at and learns from each step of the way.

**Batch Size**
The number of data samples that the machine considers in one go during training.

**Epochs**
A complete pass through the entire training dataset. The more epochs, the more the computer goes over the material to learn.

**Dataset Splits**
Dividing the dataset into parts for different uses, such as training the model and testing how well it works.

#### Code Example
```python
from transformers import (DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=64,
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()

```

## Pre-Trained Models and Transfer Learning

By using pre-trained models and the magic of transfer learning, the hard work of training an AI model from zero can be bypassed, making it easier and quicker to get the job done. By leveraging the knowledge a model distils from a large dataset, we can reduce the amount of training needed to get a performant model.

Transfer learning is a technique where a model developed for a specific task is reused as the starting point for a model on a second task. This approach leverages the knowledge gained from one problem to improve performance on another, related problem.

**Examples:**
-   If we wanted to create a plant identification app for mobile devices, we might use  [MobileNetV3(opens in a new tab)](https://paperswithcode.com/method/mobilenetv3)  and train it on a dataset containing photos of different plant species.
-   If we wanted to create a social networking spam classifier, we might use BERT and train it on a dataset containing samples of spam and not-spam text.

**Benefits**
* **Reduced Training Time**: Since the model has already learned from a large dataset, it requires less time to adapt to the new task.
* **Improved Performance**: Models that utilize transfer learning often achieve better accuracy, especially when the new task has limited data.

**How It Works**
* A pre-trained model is fine-tuned with a smaller, task-specific dataset. This involves updating the model's weights based on the new data while retaining the knowledge from the original training.
* The final layer of the model may be modified to fit the new task's requirements.


**Applications**
Transfer learning is widely used in various fields, including natural language processing (NLP) and computer vision, where pre-trained models can be adapted for specific tasks like sentiment analysis or image classification.
