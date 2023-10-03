# Custom SageMaker Project with Github Actions

A scaled-down, easy to start with, custom SageMaker Project using Github Actions for building the SageMaker Pipeline, and for deploying a real-time SageMaker endpoint.

![Overview of solution](assets/custom_sagemaker_project.png "Solution overview")

### `/build`

All the relevant pieces for a data scientist to explore the data and train a model in a notebook environment, and ultimately check the relevant pieces of code in to automatically create a reusable model training pipeline with SageMaker pipelines that registers the model with the SageMaker model registry

### `/deploy`

The build and deployment of the real-time SageMaker endpoint with data-drift and explainability monitoring. The build script is run in Github actions to create the correct parameters file, which is then used to deploy the cloudformation template
