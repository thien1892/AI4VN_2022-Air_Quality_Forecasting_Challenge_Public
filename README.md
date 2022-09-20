# Air Quality Forecasting Challenge - AIVN 2022

## 1. Overview:
- **Copy data**: You copy data to 2 folders:
    - Data train: './data-train'
    - Data public-test: './public-test/input/'
- **Train**:
    ```
    python train.py --conf_data=./CONFIG/model_thien.yml --name_model=thien1892/
    ```
- **Submit**:
    ```
    python submit.py --path_save_submit=./submit/ --path_data_test=./public-test/input/ --conf_model=./save_model/thien1892/model_save.yml
    ```
- Submit file **'./submitthien1892.zip'** --> score: 51.5746

## 2. Train

### 2.1.Args train:
- **conf_data**: default='./CONFIG/model_thien.yml', help='path of config data'
- **name_model**: default='thien/', help='name folder model to save',**keep in mind that** name_model contains '/', model will save in folder './save_model/**name_model**'
- **load_weight_imputer**: default=False, help='if true load weight was fitted', when use imputer to process missing data, it take long time. If set default **imputer_choose**=XGB, **keep_col_percent**=20, you should set **load_weight_imputer**=True, it will save time for you.
- Another Args, you can see in file **train.py**

### 2.2.Train
- cd to directory, you save code
- install requirements
```
!pip install -r requirements.txt
```
- Train: Model after train will save in path './save_model/name_model', **keep in mind that** name_model contains '/', example name_model='thienchan/'
```python
!python train.py \
 --conf_data=./CONFIG/model_thien.yml \
 --name_model=<your name model> \
 --load_weight_imputer=True
```
## 3.Submit

### 3.1. Args submit
- **path_save_submit**: Path folder save result csv
- **path_data_test**: path folder data test
- **conf_model**: when you finish your train model, 1 file yml will generate in folder save_model

**File submit zip** will be generated **./submit<name_model>.zip**

### 3.2. User model that i summited with the best score (51.5746)

```python
!python submit.py \
 --path_save_submit=./submit/ \
 --path_data_test=./public-test/input/ \
 --conf_model=./save_model/130822_12/model_save.yml
```

### 3.3. Use your train model

```python
!python submit.py \
 --path_save_submit=./submit/ \
 --path_data_test=./public-test/input/ \
 --conf_model=<Path yml file in your train model>
```