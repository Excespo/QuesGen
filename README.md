# QuesGen
This is the xlance project from 2022/10 on. It derives from the data augmentation for WebSRC, but can be more useful.

# Structure

## WebSRC data
Stored under `data/WebSRC`. This directory includes two versions of data: one reduced version and one complete version. 

`test/`: tools to test if the data is in correct format. With 
```commandline
bash ./test/test_all.sh ./reduced
```
The test environment should have `pandas` and `pytest` libraries. 

`preprocess.py`: used to produce npr masks in TIE model. Maybe useless.

`generate.py`: 
```commandline
python generate.py --root_dir ./reduced --version ques-gen
```

`dataset.py`: 

`process.sh`: run with the following command to check format and generate dataset
```commandline
bash process.sh ./reduced
```