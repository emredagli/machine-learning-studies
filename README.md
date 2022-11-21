# Machine Learning Studies

This workspace contains ML projects and studies.

## Projects and Case Studies

This section mainly contains [Machine Learning Engineer Nanodegree Program](https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189) course notes & projects.

### [Sentiment Analysis Web App Project](udacity_projects/sentiment_analysis_project)
Creating a sentiment analysis model for movie comments and building an API endpoint by using AWS SageMaker to serve the model in production env.
The deployed model produces a score between 0 (bad) and 1 (good) for an asked movie comment.
* Dataset: [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
  * X: [comment]  
  * y:[score] -> 0, 1 values
* Model: LSTM Text Classifier using Pytorch
* Deployment: Sagemaker Endpoint & (Lambda Function + API Gateway)


### Case Studies

The following ML case studies were studied and implemented as guided on the [Udacity repository](https://github.com/udacity/ML_SageMaker_Studies)

#### Population Segmentation
Training and deploying unsupervised models (PCA and k-means clustering) to group US counties by similarities and differences.
Visualizing the trained model attributes and interpreting the results.

#### Payment Fraud Detection
Training a linear model to do credit card fraud detection. 
Improving the model by accounting for class imbalance in the training data and tuning for a specific performance metric.

#### Deploying Custom Models via Amazon SageMaker
Designing and training a custom PyTorch classifier by writing a training script. 
This is an especially useful skill for tasks that cannot be easily solved by built-in algorithms.

#### Time-Series Forecasting
Formatting time series data into context (input) and prediction (output) data, and training the built-in algorithm, DeepAR; this uses an RNN to find recurring patterns in time series data.

[//]: # (### [Plagiarism Detection Project]&#40;udacity_projects/plagiarism_detection_project&#41;)

[//]: # (Building a plagiarism detector model that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar that text file is to a provided source text. )

[//]: # (Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.)

### [ML Based Real Estate Appraisal Project](udacity_projects/ml_based_real_estate_appraisal_project)
Building & deploying a real estate appraisal model by using a real dataset. Note: The project has not been finished yet.


