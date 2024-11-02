# phmd: Prognosis Healgth Management Datasets tool

The library `phmd` is designed to facilitate access to datasets in the context of industrial prognosis and health management (PHM). The main goal is to streamline automated processing, enabling researchers and practitioners to obtain, manipulate, and analyze data relevant to their work in an easy maner. 

In the realm of predictive maintenance and fault diagnosis, the ability to efficiently search and access relevant datasets is crucial for developing effective machine learning models. The phmd library provides a straightforward method to search through a comprehensive collection of datasets based on specific features or criteria. This functionality allows users to quickly locate datasets that match their needs, enhancing the efficiency of their research and development processes.

This library provides tools for data search, downloading, and loading without concerns about the locations or source formats of the datasets. In a nutshell, the library provides a unified interface for accessing an important number of datasets from diferent sources and heterogeneous format, thereby reducing significantly the time and effort required to prepare datasets for machine learning and other analytical tasks.


### Software functionalities

The main features of the developed software can be summarized as follows:

- **Listing and search**. The library allows users to search for datasets tailored to specific purposes. Each dataset is associated with a set of meta-attributes facilitating the selection of the most suitable dataset for a particular research objective.
- **Automatic download**. The download process is automatically triggered when a dataset is required to load only if it has not been previously downloaded.   
- **Loading**. One of the most important features of this library is its dataset loading capability. The loading functionality abstracts away from source format in which the dataset is originally provided, making it agnostic to the user and allowing them to focus directly on data analysis after loading. The loading functionality accepts the dataset name and the task type as parameters. Based on the specified task, the target variable is automatically computed, and the relevant features are filtered accordingly.


### Installation

You can install `phmd` with pip:

```
pip install phmd
```

or clone this repository and install it in editable mode:

```
git clone https://github.com/dasolma/phmd
cd phmd
pip install -e .
```


### Examples of usage

#### Search datasets

The method *search* of the class *Dataset* allow list of the dataset avaiable or search for datasets that match any criteria.

For example searching for datasets that contain vibration data, users can identify various datasets across different domains and applications, such as mechanical or manufacturing contexts. The following example illustrates how to perform a dataset search using the phmd library, showcasing the relevant details, including dataset names, domains, application areas, task names, and the nature of the data and features provided.

```
>>> import phmd
>>> datasets.Dataset.search(features='vibra')

name    domain        nature      app     task name [target] data nature  features
------- ------------- ----------- ------- ------------------ ------------ --------
CWRU    Mechanical    time-series Bearing Diagnosis [fault]  time-series vibration
DFD15   Manufacturing time-series Drill   Diagnosis [fault]  time-series vibration
DFD15   Manufacturing time-series Drill   Stage  [stage]     time-series vibration
...
...
...
UPM23   Mechanical    time-series Bearing Diagnosis [fault]  time-series vibration
XJTU-SY Mechanical    time-series Bearing Prognosis [rul]    time-series vibration
XJTU-SY Mechanical    time-series Bearing Diagnosis [fault]  time-series vibration
```

The `search` method allows filtering datasets by various metadata fields:

- **name**: Name of the dataset (e.g., CWRU, UOC18, etc.)
- **domain**: Domain of application
- **nature**: Nature of the data (time-series or features)
- **application**: Application of the dataset (e.g., battery, gear, building)
- **task**: Task name or target variable (e.g., diagnosis, prognosis, rul, fault, etc.)
- **features**: Type of features in the dataset (e.g., vibration, current, etc.)
- **publisher**: Publisher name (e.g., NASA, PHM Society, etc.)


### Get information of a dataset

Understanding the specifics of a dataset is vital for effective analysis and model development. The phmd library simplifies this process by providing a straightforward way to retrieve detailed information of the dataset collection. By utilizing the Dataset class, users can obtain comprehensive descriptions, system information, features, tasks, resources, and references associated with a particular dataset.

In the following example, we demonstrate how to access detailed information for the well-known CWRU (Case Western Reserve University) dataset, which focuses on bearing fault diagnosis. The output includes essential details such as the dataset's description, the types of sensors used, the nature of the data, and storage requirements. This allows users to evaluate the dataset's suitability for their specific applications and research needs.

```
>>> from phmd import datasets
>>> ds = datasets.Dataset("CWRU")
>>> print(ds.describe())

Description
===========
In this renowned dataset, experiments were conducted utilizing 
a 2 HP Reliance Electric motor, where acceleration data was 
measured at locations both near to and remote from the motor...

Designation: Bearing Fault Diagnostic
Publisher: Case Western Reserve University
Domain: Mechanical component
Application: Bearing
License: CC BY-SA 4.0

System info
===========
1. type    : Rotatory machine :  bearing
2. sensors : Voltmeter, ammeter and thermocouple sensor suite
3. bearing : 6205-2RSL JEM SKF deep-groove ball bearing (and NTN equivalent)

Features
========
BA :
   description : base accelerometer data
   type        : vibration
DE :
   description : drive end accelerometer data
   type        : vibration
FE :
   description : fan end accelerometer data
   type        : vibration

Tasks
=====
Diagnosis :
   features            : DE
   identifier          : unit
   min_ts_len          : 63788
   nature              : time-series
   num_units           : 161
   target              : fault
   target_distribution : 0.24,0.51,0.23,0.021
   target_labels       : IR,OR,BA,NO
   type                : classification:multiclass

Resources
=========
1. storage :
   a) zipped   : 246MB
   b) unzipped : 689MB
   c) RAM :
      Data set (full) : 5.2GB
2. load time (SSD disk) :
   a) unzipped :
      Data set (full) : 3s
   a) zipped :
      Data set (full) : 7s

References
==========
citation        : K.A. Loparo, Bearings vibration data set. 
                  The Case  Western Reserve University Bearing 
                  Data Center. https://engineering.case.edu
manual download : https://engineering.case.edu/bearingdatacenter
```

### Load a dataset

Loading a dataset is a crucial step in any data analysis or machine learning workflow. The phmd library streamlines this process, allowing users to effortlessly load datasets and access relevant tasks and features. In this example, we demonstrate how to load the CWRU dataset.

By initializing the Dataset class with the dataset name, users can retrieve specific tasks associated with the dataset. In this case, we access the 'fault' task for analysis. The subsequent command loads the first fold subsets associated with this task. 


```
>>> ds = datasets.Dataset("CWRU")
>>> task = ds['fault']
>>> task.method = 'features'
>>> sets = task[0]

Dataset CWRU already downloaded and extracted
Remember to cite the original publisher dataset:
	@misc{caseBearingData,
		author = {},
		title = {{B}earing {D}ata {C}enter | {C}ase {S}chool of {E}ngineering 
		         {C}ase {W}estern {R}eserve {U}niversity --- engineering.case.edu},
		howpublished = {\url{https://engineering.case.edu/bearingdatacenter}},
		year = {},
		note = {[Accessed 08-04-2024]},
	}
You can download the dataset manually from: https://engineering.case.edu/bearingdatacenter

** If you find this tool useful, please cite our SoftwareX paper.

Reading data: 100%|██████████| 161/161 [00:03<00:00, 47.43it/s]
INFO:root:Read in 5.96511435508728 seconds
INFO:root:It is possible stratified split? True
INFO:root:Read 3 sets: train,val,test
INFO:root:Columns: DE,fault,unit
INFO:root:Train shape: (28567988, 3)
INFO:root:Val shape: (7804030, 3)
INFO:root:Test shape: (979629, 3)
```

The output informs the user about the status of the dataset, indicating whether it has already been downloaded and extracted. It also provides a citation format for referencing the dataset in future work, along with a manual download link. Additionally, the process logs detailed information about data reading, allowing users to monitor performance and resource usage during the loading process.

The `Task` class allows for setting various attributes to configure the reading of fold sets:

- **folds** *(Int)*: Defaults to 5 or the maximum number of folds depending on the number of units present in the dataset.
- **preprocess** *(Str)*: Defaults to *None*. Other valid options include *std* for standardization and *norm* for normalization of data.
- **normalize_output** *(Bool)*: Defaults to *False*. In tasks with continuous targets, such as RUL, normalizing the output is common. When set to *True*, the target column will be normalized.
- **test_pct** *(Float)*: Defaults to 0.3, specifying the percentage of data used for the test set.
- **return_test** *(Bool)*: Defaults to *True*. If set to *False*, only the validation and training sets are returned.
- **random_state** *(Int)*: Sets the random seed used to split the data. Changing this value results in a different data split.

## ACKNOWLEDGMENT

This work has been supported by Grant PID2019-109152GBI00/AEI/10.13039/501100011033 (Agencia Estatal de Investigacion), Spain and by the Ministry of Science and Education of Spain through the national program "Ayudas para contratos para la formacion de investigadores en empresas (DIN2019)", of State Programme of Science Research and Innovations 2017-2020.



