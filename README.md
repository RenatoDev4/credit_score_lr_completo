# Credit Score LR

DEMO application of a machine learning model to assess the probability of a customer defaulting on a credit card application


### Requirements to run project:
-----------
- Python 3.10+
- Pandas
- Numpy
- Category-Encoders
- Statsmodels
- Optbinning
- Dython
- Scikit-Learn
- Streamlit

``` bash
$ pip install -r requirements.txt
```
and

``` bash
$ source src/data/setup.sh
```

### You can run with docker:
-----------
``` bash
docker pull renatodev4/credit_score_lr
```

Using this command a docker container will be started with the project running with streamlit in your local machine.


### Start the project manually, run:
------------

``` bash
$ streamlit run src/data/dashboard.py
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw
    │
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download and generate data
    │   │   └── dashboard.py
    │   │
    │   ├── features       <- Config file
    │   │   ├── config.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions 
    │   │   ├── predict_model.py
    │   │
    └───── setup.sh <- Create the environment variable to run the project

## Made by:

Project made by **Renato Moraes** for his portfolio.<br>
You can check the Data Science project running at: https://github.com/RenatoDev4/credit_score_lr_completo


--------