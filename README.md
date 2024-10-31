# Report and code by Thomas Selin related to project 1 (module 2) in the university course `Multivariate data analysis` at `Mälardalens University`

An autoencoder machine learning model for anomaly detection in an IT service component (microservice) was created.

See the [Project Report](Project_report.pdf) for a detailed report on the execution, result and conclusions of the project.

## Notes:

- PCA was performed on the dataset in earlier project, but as it didn't help much in dimensionality reduction and would lead to a less explainable model, that result was not used when creating the final model.

- The dataset is not included as consists of somewhat sensitive company internal data. See the [Project Report](Project_report.pdf) for some further insight into the data.

## Improved version
An improved version of the code was added after project completion. The improvements include:
- uses the Huber loss function
- has early stopping 
- has model checkpoint
- has evaluation on the test set

You can find the improved version in the `improved_model_creation.py` file.

## Potential further improvements

- Adding small Gaussian noise to the input (denoising autoencoder) to prevent overfitting. For autoencoders this would likely give better result compared to using dropout.
