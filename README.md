# PredM
Companion repository to the paper [Enhancing Siamese Neural Networks through Expert Knowledge for Predictive Maintenance](http://www.wi2.uni-trier.de/shared/publications/2020_ECML-IoTStreams_SiameseNeuralNetwork-for-PredictiveMaintenance_Preprint.pdf). \
Please note, that the MS-SNNs approach is here referred to as CaseBasedSimilarity (CBS) and CNN2D + MAR is called cnn2dWithAddInput.
The implementation of some components is based on the one presented in [NeuralWarp](https://arxiv.org/abs/1812.08306) ([GitHub](https://github.com/josifgrabocka/neuralwarp)).

## Supplementary Resources
* The sub directory [supplementary\_resources](https://github.com/PredM/SiameseNeuralNetwork/tree/master/supplementary_resources) of this repository contains additional information about the data sets used and the architecture of the CNN2D + MAR model.
* An [overview](https://seafile.rlp.net/f/fa6624358ff04627b9a8/) of all relevant conducted experiments
* The [detailed logs](https://seafile.rlp.net/d/81938c117f444eddbda4/) for each of those experiments 
* The [raw data](https://seafile.rlp.net/d/cd5590e4e9d249b2847e/) recorded with this [simulation factory model](https://iot.uni-trier.de) used to generate the training and evaluation data sets.
* The [preprocessed data set](https://seafile.rlp.net/d/69434c0df2c3493aac7f/) we used for the evaluation.

## Quick start guide: How to start the model?
1. Clone the repository
2. Download the [preprocessed data set](https://seafile.rlp.net/d/69434c0df2c3493aac7f/) and move it to the _data_ folder
3. Navigate to the _neural_network_ folder and start the training and test procedure via _python TrainAndTest.py > Log.txt_

## Requirements
Used python version: 3.6.X \
Used packages: See requirements.txt

## Hardware
<table>
    <tr>
        <td>CPU</td>
        <td>2x 40x Intel Xeon Gold 6138 @ 2.00GHz</td>
    </tr>
    <tr>
        <td>RAM</td>
        <td>12 x 64 GB Micron DDR4</td>
    </tr>
       <tr>
        <td>GPU</td>
        <td>8 x NVIDIA Tesla V100 32 GB GPUs</td>
    </tr>
</table>

## General instructions for use
* All settings can be adjusted in the script Configuration.py, 
whereby some rarely changed variables are stored in the file config.json, which is read in during the initialization.
* The hyperparameters of the neural networks can be defined in the script Hyperparameter.py or can be imported from a file in configuration/hyperparameter_combinations/ (this can also be changed in the configuration).
* For training, the desired adjustments should first be made at the parts mentioned above and then the training can be started by running Training.py.
* The evaluation of a trained model on the test dataset can be done via Inference.py. 
To do this, the folder which contains the model files, must first be specified in the configuration. 
* For executing the real-time data processing using RealTimeClassification.py first a kafka server must be configured and running. Also the topic names and mappings to prefixes must be set correctly.
* The data/ directory contains all required data. Central are the pre-processed training data in data/training_data/ and the trained models in data/trained_models/. 
A detailed description of what each directory contains is given in corresponding parts of the configuration file. 

## Software components
The following section gives an overview of the packages, directories and included Python scripts in this repository. 

### analytic_tools

| Python script | Purpose |
| ---      		|  ------  |
|ExampleCounter.py | Displays the example distribution in the training data and the case base. |
|ExtractCases.py|Automatically determines the time intervals at which simulated wear is present on one of the motors and exports these into to a text file.|
|LightBarrierAnalysis.py| Used for manual determination of error case intervals for data sets with light barrier errors.|
|PressureAnalysis.py|Used for manual determination of error case intervals for data sets with simulated pressure drops.|
|CaseGrouping.py| Is used to generate an overview of the features used for each error case and to create a grouping of cases based on this.|

### archive
The archive contains currently unused code fragments that could potentially be useful again, old configurations and such.

### baseline
| Python script | Purpose |
| ---      		|  ------  |
|BaselineTester.py| Provides the possibility to apply other methods for determining similarities of time series, e.g. DTW, to the data set. |

### case_based_similarity
| Python script | Purpose |
| ---      		|  ------  |
|CaseBasedSimilarity.py| Contains the implementation of the case-based similarity measure (CBS). |
|Inference.py| Evaluation of a CBS model based on the test data set. |
|Training.py| Used for training a CBS model.|

### configuration
| Python script | Purpose |
| ---      		|  ------  |
|Configuration.py|The configuration file within which all adjustments can be made.|
|Hyperparameters.py| Contains the class that stores the hyperparameters used by a single neural network.|

### data_processing
| Python script | Purpose |
| ---      		|  ------  |
|CaseBaseExtraction.py| Provides extraction of a case base from the entire training data set.|
|DataImport.py|This script executes the first part of the preprocessing. It consists of reading the unprocessed sensor data from Kafka topics in JSON format as a *.txt file (e.g., acceleration, BMX, txt, print) and then saving it as export_data.pkl in the same folder. This script also defines which attributes/features/streams are used via config.json with the entry "relevant_features". Which data is processed can also be set in config.json with the entry datasets (path, start, and end timestamp). |
|DataframeCleaning.py|This script executes the second part of the preprocessing of the training data. It needs the export_data.pkl file generated in the first step. The cleanup procedure consists of the following steps: 1. Replace True/False with 1/0, 2. Fill NA for boolean and integer columns with values, 3. Interpolate NA values for real valued streams, 4. Drop first/last rows that contain NA for any of the streams. In the end, a new file, called cleaned_data.pkl, is generated.|
|DatasetCreation.py|Third part of preprocessing. Conversion of the cleaned data frames of all partial data sets into the training data.|
|DatasetPostProcessing.py | Additional, subsequent changes to a dataset are done by this script.|
|RealTimeClassification.py|Contains the implementation of the real time data processing.|
        
### fabric_simulation
| Python script | Purpose |
| ---      		|  ------  |
|FabricSimulation.py|Script to simulate the production process for easier development of real time evaluation.|

### logs
Used to store the outputs/logs of inference/test runs for future evaluation.

### neural_network
| Python script | Purpose |
| ---      		|  ------  |
|BasicNeuralNetworks.py| Contains the implementation of all basic types of neural networks, e.g. CNN, FFNN.|
|Dataset.py|Contains the class that stores the training data and meta data about it. Used by any scripts that uses the generated dataset|
|Evaluator.py|Contains an evaluation procedure which is used by all test routines, i.e. SNNs, CBS and baseline testers.|
|Inference.py|Provides the ability to test a trained model on the test data set.|
|Optimizer.py|Contains the optimizer routine for updating the parameters during training. Used for optimizing SNNs as well as the CBS.|
|SimpleSimilarityMeasure.py|Several simple similarity measures for calculating the similarity between the enbedding vectors are implemented here.|
|SNN.py|Includes all four variants of the siamese neural network (classic architecture or optimized variant, simple or FFNN similiarty measure).|
|TrainAndTest.py| Execution of a training followed by automatic evaluation of the model with best loss.|
|Training.py| Used to execute the training process.|

## Compatibility
Due to the high amount of different models and configuration options, not all components can be used together. 
The following table shows the current compatibility status of the different models with the execution variants.
Also please note that the real time classification is still under development and may not work currently.

<table>
    <tbody>
      <tr>
        <td></td>
        <td>SNN</td>
        <td></td>
        <td></td>
        <td></td>
        <td>CBS</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td>Standard</td>
        <td></td>
        <td>Fast</td>
        <td></td>
        <td>Standard</td>
        <td></td>
        <td>Fast</td>
        <td></td>
      </tr>
      <tr>
        <td>Encoder</td>
        <td>Simple</td>
        <td>FFNN</td>
        <td>Simple</td>
        <td>FFNN</td>
        <td>Simple</td>
        <td>FFNN</td>
        <td>Simple</td>
        <td>FFNN</td>
      </tr>
      <tr>
        <td>CNN</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
      </tr>
      <tr>
        <td >RNN</td>
        <td >Working</td>
        <td >Working</td>
        <td>Working</td>
        <td >Working</td>
        <td >Working</td>
        <td >Working</td>
        <td>Working</td>
        <td>Working</td>
      </tr>
      <tr >
        <td>cnn2dwithaddinput</td>
        <td>Working</td>
        <td>Not working / Error</td>
        <td>Not implemented yet</td>
        <td>Not implemented yet</td>
        <td >Not planned / necessary</td>
        <td >Not planned / necessary</td>
        <td >Not planned / necessary</td>
        <td >Not planned / necessary</td>
      </tr>
      <tr>
        <td >cnn2d</td>
        <td>Working</td>
        <td>Working</td>
        <td>Working</td>
        <td >Working</td>
        <td>Not working / Error</td>
        <td >Not working / Error</td>
        <td >Not working / Error</td>
        <td >Not working / Error</td>
      </tr>
    </tbody>
</table>
