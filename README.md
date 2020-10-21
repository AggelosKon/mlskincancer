# mlskincancer
Machine Learning on Skin Cancer

The Dataset can be found at: https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign

Data Set folders structure initially must be:

data

	| -> train
	
		|-> benign
		|-> malignant
		
		
	| -> test
		|-> benign
		|-> malignant

Caution: Make sure that all .ipynb files and data folder described above are in the same folder!

Run Populate_DataSet.ipynb to create a 4 times greater data set with images edited
The new data set will be stored in a folder named 'augmented_dataset' which will contain
* train set (70% of the augmented data set)
* validation set (10% of the augmented data set)
* test set (20% of the augmented data set)
all folders contain images of benign and malignant cutaneous tumors.

Run main.ipynb and choose among the models:
1) MultiLayer Perceptron Neural Network
2) Concolutional Neural Network
3) Stochastic Gradient Descent
4) Support Vector Machine
