# NET_topside_model
Neural network model of Electron density in the Topside ionosphere (NET) model files. Version 1.0. 

## Python dependencies to run the model:
* python v>3.8.
* numpy
* datetime
* pandas
* h5py
* joblib
* keras v>2.8.0
* tensorflow v>2.8.0
* scikit-learn v=1.2.0

 The easiest way to run the model is to create a special virtual environment using conda for these package versions.
 
## Example
The example and explanations are provided in the jupyter notebook

## The model inputs are:
* Altitude
* Geographic latitude
* Geographic longitude
* Magnetic latitude
* Magnetic longitude
* Magnetic local time
* Day of year
* SYM-H index
* P10.7 solar flux index
* Kp index
