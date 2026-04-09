
## Introduction
<p>
  Predicting stock prices is a cumbersome task as it does not follow any specific pattern. Changes in the stock prices are purely based on supply and demand during a period of time. In order to learn the specific characteristics of a stock price, we can use algorithm to identify these patterns through machine learning. One of the most well-known networks for series forecasting is LSTM (long short-term memory) which is a Recurrent Neural Network (RNN) that is able to remember information over a long period of time, thus making them extremely useful for predicting stock prices. RNNs are well-suited to time series data and they are able to process the data step-by-step, maintaining an internal state where they cache the information they have seen so far in a summarised version. The successful prediction of a stock's future price could yield a significant profit.
</p>

## Aim
<p> 
  To predict stock prices according to real-time data values fetched from API.
</p>

## Objective
<p>
  The main objective of this project is to develop a web application that can predict stock price based on real-time data.  
</p>

## Project Scope
<p>
  The project has a wide scope, as it is not intended to a particular organization. This project is going to develop generic software, which can be applied by any businesses organization. Moreover it provides facility to its users. Also the software is going to provide a huge amount of summary data. 
</p>
  
#
## Prerequisites:
```bash
  Django==3.2.6
  django-heroku==0.3.1
  gunicorn==20.1.0
  matplotlib==3.5.2
  matplotlib-inline==0.1.3
  numpy==1.23.0
  pandas==1.4.1
  pipenv==2022.6.7
  plotly==5.9.0
  requests==2.28.1
  scikit-learn==1.1.1
  scipy==1.8.1
  seaborn==0.11.2
  sklearn==0.0
  virtualenv==20.14.1
  virtualenv-clone==0.5.7
  yfinance==0.1.72
```

## Project Installation:
**STEP 1:** Clone the repository from GitHub.
```bash
  git clone https://github.com/sreenu1232/stockmarkettrendprediction.git
```

**STEP 2:** Create a virtual environment
(For Windows)
```bash
  python -m venv virtualenv
```
(For MacOS and Linux)
```bash
  python3 -m venv virtualenv
```

**STEP 3:** Activate the virtual environment.
(For Windows)
```bash
  virtualenv\Scripts\activate
```
(For MacOS and Linux)
```bash
  source virtualenv/bin/activate
```

**STEP 4:** Install the dependencies.
```bash
  pip install -r requirements.txt
```

**STEP 5:** Migrate the Django project.
(For Windows)
```bash
  python manage.py migrate
```
(For MacOS and Linux)
```bash
  python3 manage.py migrate
```

**STEP 6:** Run the application.
(For Windows)
```bash
  python manage.py runserver
```
(For MacOS and Linux)
```bash
  python3 manage.py runserver
```
