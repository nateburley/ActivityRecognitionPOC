# Activity Recognition POC

## Summary
A simple proof-of-concept to test the efficacy of using a lightweight, simple model architecture, such as SciKit Learn or XGBoost, to classify activities being performed on smart watches.

## Purpose
Apple's new Core ML framework allows models to be trained on desktop computers or in the cloud, and deployed to the Apple Watch for edge inference. It thus follows that, if we could build a model to passively (and anonymously) detect activities in the background, we could perform services like content recommendation. For example, if we detect that someone is taking a walk, Headspace could recommend the content piece "Take a Mindful Walk". However, due to the limited compute resources and power supply on an Apple Watch, the model needs to be accurate, while performing inference quickly and at low cost. 

## Data
This proof-of-concept for lightweight activity recognition uses Fordham University's open-source WISDM Dataset, a collection of labeled accelerometer data from twenty-nine users performing daily activities such as walking, jogging, climbing stairs, sitting, and standing. The raw time series sensor data is then aggregated into examples that summarize the user activity over 10-second intervals. More information can be found in their paper <a href="https://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf">here</a>.
