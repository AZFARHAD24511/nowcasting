# Nowcasting

## Abstract
Numerous forecasting methodologies have been proposed in economic literature, with VAR, ARMA, ARIMA, and related models standing out for time series analysis. However, accessing the requisite data for these models, especially at high frequencies such as daily, poses challenges. Even when available, timely publication of such data is often lacking. Consequently, decision-makers and economic policymakers remain unaware of daily and weekly fluctuations in the country's economy. This project aims to explore whether Google Trends data, which is published monthly, daily, and even hourly, holds predictive power for foreign trade—a crucial target variable. Daily Google Trends data spanning from 2006 to October 2023 was extracted using the Google Trends API. Concurrently, foreign trade data was sourced from the German Federal Government Statistics Office website.
## Problem statement
1. Presenting a novel deep learning approach aimed at predicting the foreign trade trends of Germany.
2. Utilizing Google Trends information to anticipate foreign trade patterns, especially in scenarios lacking access to real-time economic data.

## Method<a name="method"></a>
The methodology of our project involves leveraging the power of the Long Short-Term Memory (LSTM) model. This algorithm is subset of recurrent neural networks (RNNs) tailored to address the vanishing gradient problem commonly encountered in traditional RNNs.

Utilizing LSTM proves advantageous, particularly in analyzing economic data and time-related information, due to its adeptness in discerning patterns and relationships within sequential data over time.

In our study, I applied the LSTM model to scrutinize economic data across time periods. Our approach entailed configuring a specific architecture comprising 49 features and one target variable within the LSTM layer. We examined sequences of daily data, each spanning a length of 30 days or more, with the objective of capturing enduring relationships within the dataset.


### Data Collection<a name="data-collection"></a>
We got information from Google Trends using a special computer program called R. This program helped us find data related to business and money based on certain words. You can learn more about this program by visiting this [website](https://github.com/PMassicotte/gtrendsR). We collected Google Trends data every day from 2006 to October 2023 using a tool called Google Trends API. We also gathered information about international trade from the official website of Germany's statistics office. These are the words we looked at on Google Trends, listed below. I've simplified the names from x1 to x49.
|   | 1                   | 2                  | 3                | 4                    | 5                   | 6                   | 7                  |
|---|---------------------|--------------------|------------------|----------------------|---------------------|---------------------|--------------------|
| 1 | Aktienhandel       | Aktienkurse        | Aktienmarkt      | Aktienmarkt heute   | Altersvorsorge     | Anlageberatung     | Arbeitslosigkeit   |
| 2 | Austauschjahr      | BIP                | Besteuerung      | Einkommenssteuer    | Federal Reserve    | Finanzbericht      | Finanzkrise        |
| 3 | Finanzmärkte       | Forex -Handel      | Geldpolitik      | Handelsdefizit      | Haushaltsdefizit   | Hypothekenzinsen   | Immobilienmarkt    |
| 4 | Inflation          | Investition        | Investment Banking| Kapitalanlage       | Konjunktur         | Kredit -Score      | Kreditberatung     |
| 5 | Kryptowährung      | Marktanalyse       | Marktforschung   | Rezession           | Sparkonto          | Steuerabzüge       | Steuerreform       |
| 6 | Verbraucher        | Wechselkurse       | Welthandel       | Wirtschaft          | Wirtschaftslage    | Wirtschaftsnachrichten | Wirtschaftspolitik |
| 7 | Wirtschaftsunternehmen | Wirtschaftswachstum | Zentralbank | Zinsen              | Zwangsvollstreckung | wirtschaftliche Entwicklung | Ökonomisch |


Also the target variable "y" is German daily trade data
### Tuning hyperparameters<a name="tuning"></a>
Given the various layers present in the LSTM model and the array of hyperparameters it encompasses, it becomes imperative to ascertain both the ideal number of layers and the optimal hyperparameters. Therefore, my focus has been on fine-tuning both the neuron count and dropout rate hyperparameters to achieve optimal performance.
### evalueation model<a name="evaluate"></a>
The model's evaluation using **$R^2$** indicates that 97% of the changes in the target are explained by the features.

## References<a name="references"></a>

1. PMassicotte (n.d.). gtrendsR: An R Package for Downloading Google Trends Data. GitHub. Retrieved April 19, 2024, from https://github.com/PMassicotte/gtrendsR.
2. Foreign trade. (2023, 11 25). Retrieved from website of The Federal Statistical Office: https://www.destatis.de/EN/Home/_node.html


## Implemnetation of LSTM
### 1-Importing libraries

### 2-loading datasets and reporting dataset information
The code imports a dataset from a GitHub repository hosted by the user with the specified address_web. Before processing, it displays the data types of the dataset columns. Since the data is time-series data, the code sets the 'date' column as the index and removes it from the dataset columns. Additionally, to prepare the dataset for machine learning tasks, the code converts all data types to float. After these changes, it displays the data types of the dataset columns again.
### 4- Splitting dataset to train and test and reporting thier shape or dimensions
### 5-Standardizaion features
### 6-creating sequences
In this code snippet, we're preparing our data for model training. We define the length of each sequence of data to be 10 time steps and specify that we want to process our data in batches of 50 sequences at a time. Using the `preprocessing.timeseries_dataset_from_array` function, we create training, valid, and test datasets. For each dataset, we provide the input data (`Xtrain`, `Xvalid`, `Xtest`) and the corresponding target labels (`ytrain`, `y_valid`, `ytest`). We set the sequence length and batch size for each dataset to the values we defined earlier. This process helps us organize our data into suitable formats for training and evaluating our model.

### 7- model tuning
I want to implement an LSTM model. I assume that this model has three hidden layers. Of course, I came to this conclusion based on experience. Now I assume that there are 96, 64 and 32 neurons in the first, second and third layer respectively. I also assume I set the dropout to be 0.5. Assuming this, my pattern will be as follows.

### 7-2-Tuning neurons and dropout
The provided code defines a pipeline class that automates the process Machin lerning:\
This code is about training a machine learning model using a special type of neural network called Long Short-Term Memory (LSTM). The model is trained to predict future values based on past data.\
1. The data is imported from a website and preprocessed.
2. It is split into training, validation, and testing sets.
3. The data is standardized to ensure all features have the same scale.
4. The data is organized into sequences, which are fed into the LSTM model for training.
5. The model architecture is defined with multiple LSTM layers followed by a dropout layer to prevent overfitting.
6. The model is trained for multiple combinations of LSTM units, and the performance is evaluated using the 𝑅² score, which measures how well the model predicts the data.
7. Finally, the results are compared to find the best combination of LSTM units and dropout rate for the model.

In this step, I tried to automatically extract all the pipeline, but only tuned two hyperparameters. This is really the number of neurons and dropout in the three hypothesized hidden layers of LSTM.
```
lstm_units_values = [[16, 64, 32],[16, 64, 16],[16, 64, 8],[16, 32, 32],[16, 32, 16],[16, 32, 8]]
dropout_values = [0.01, 0.015, 0.02]
```
the result can be seen in the table below that the **$R^2$** number is slightly more than 98%.
As you can see, with the number of neurons in the first to third layers and the following Dropout, a higher number can be achieved.With LSTM_units:**[16, 64, 16]**  and Dropout: **0.015** the **$R^2$** score is 97%

| Pipeline                | LSTM_units   | Dropout | R2_score |
|-------------------------|--------------|---------|----------|
| Pipe_Line_2_Dropout_0.01| [16, 64, 16] | **0.010**   | **0.969108** |
| Pipe_Line_1_Dropout_0.015| [16, 64, 32] | 0.015   | 0.954294 |
| Pipe_Line_2_Dropout_0.015| [16, 64, 16] | 0.015   | 0.964002 |
| Pipe_Line_1_Dropout_0.02 | [16, 64, 32] | 0.020   | 0.937698 |
| Pipe_Line_4_Dropout_0.015| [16, 32, 32] | 0.015   | 0.926979 |
| Pipe_Line_4_Dropout_0.02 | [16, 32, 32] | 0.020   | 0.931688 |
| Pipe_Line_5_Dropout_0.015| [16, 32, 8]  | 0.015   | 0.938163 |
| Pipe_Line_5_Dropout_0.01 | [16, 32, 8]  | 0.010   | 0.894652 |
| Pipe_Line_1_Dropout_0.01 | [16, 64, 32] | 0.010   | 0.916466 |
| Pipe_Line_4_Dropout_0.01 | [16, 32, 32] | 0.010   | 0.906207 |
| Pipe_Line_2_Dropout_0.02 | [16, 64, 16] | 0.020   | 0.961831 |
| Pipe_Line_3_Dropout_0.02 | [16, 64, 8]  | 0.020   | -0.341800 |
| Pipe_Line_3_Dropout_0.015| [16, 64, 8]  | 0.015   | -0.367244 |
| Pipe_Line_3_Dropout_0.01 | [16, 64, 8]  | 0.010   | -0.309727 |
| Pipe_Line_5_Dropout_0.02 | [16, 32, 8]  | 0.020   | 0.887368 |

## final reuslt
Of course, LSTM  also has other hyperparameters, and due to the time-consuming nature of their testing, I only limited myself to these two. Since the result of this model was associated with high **$R^2$** , I considered this as a final model.
## Future research:
One of the things that should be done in future research is tuning the number of optimal layers for the LSTM model. Little work has been done in this regard.
