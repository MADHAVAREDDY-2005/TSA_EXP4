# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 23-09-2025
## Name: K MADHAVA REDDY
## Register. No: 212223240064



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load avocado dataset
data = pd.read_csv('/content/avocado.csv')

# Convert Date to datetime and sort
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Use 'Total Volume' for modeling
X = data['Total Volume']

# Plot original avocado total volume data
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(data['Date'], X)
plt.title('Original Avocado Total Volume Data')
plt.xlabel("Date")
plt.ylabel("Total Volume")
plt.show()

# ACF and PACF of original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF (Total Volume)')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF (Total Volume)')

plt.tight_layout()
plt.show()

# Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Avocado Total Volume')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Avocado Total Volume')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

OUTPUT:
<img width="1006" height="537" alt="image" src="https://github.com/user-attachments/assets/819f1593-3091-4100-a300-cb75aeccc603" />

<img width="1188" height="575" alt="image" src="https://github.com/user-attachments/assets/192d9648-4440-4810-b85e-3eccc83b6f43" />


### SIMULATED ARMA(1,1) PROCESS:
<img width="987" height="529" alt="image" src="https://github.com/user-attachments/assets/4cd95159-90d5-4468-ae2e-7d093c16adb7" />



### Partial Autocorrelation
<img width="996" height="512" alt="image" src="https://github.com/user-attachments/assets/12961d27-0cc8-4660-92d8-aa176096d1c6" />

### Autocorrelation
<img width="990" height="513" alt="image" src="https://github.com/user-attachments/assets/0a3a48f4-aae9-4d41-9916-827e82130eda" />



### SIMULATED ARMA(2,2) PROCESS:
<img width="985" height="525" alt="image" src="https://github.com/user-attachments/assets/8b6d3444-b3f1-4bbc-830f-df22aabc6e50" />

### Partial Autocorrelation
<img width="996" height="508" alt="image" src="https://github.com/user-attachments/assets/dd3c1277-c424-4d3d-ad04-9e7f04b48bb9" />



### Autocorrelation
<img width="1001" height="518" alt="image" src="https://github.com/user-attachments/assets/081ad85a-de80-4bfa-81db-671d5b4eb19b" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
