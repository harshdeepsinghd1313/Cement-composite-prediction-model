# Cement-composite-prediction-model
## AIM OF THE PROJECT 
### To Predict Ultimate compressive strenght of Cement Composite mixture using Machine Learning Technique.

## Summary of the Project
Objective:
The primary goal of this project is to predict the ultimate compressive strength of cement composite mixtures using machine learning techniques. This prediction helps in optimizing cement formulations for better structural integrity in construction applications. By leveraging data-driven insights, the project aims to enhance the efficiency of material selection and ensure cost-effective yet durable concrete compositions.

Dataset:
The dataset consists of multiple features representing different components of cement composite mixtures, such as:
Cement
Water
Aggregates
Additives
These variables significantly influence the compressive strength of the final mixture. Proper data preprocessing was performed, including handling missing values, feature scaling, and data normalization.

Techniques Applied:
To achieve accurate predictions, the following machine learning techniques were used:
✅ Linear Regression: Implemented as a baseline model to understand relationships between input features and output strength.

✅ Artificial Neural Network (ANN): Applied a deep learning model to capture complex non-linear patterns within the dataset. ANN was optimized through hyperparameter tuning, including adjustments in the number of layers, neurons, activation functions, and learning rate.

✅ MinMaxScaler: Used to normalize the input features, ensuring that all variables contributed equally to model learning.

✅ Model Evaluation: Metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score were used to assess model performance.

Performance & Results:
The Artificial Neural Network (ANN) model outperformed Linear Regression, demonstrating a higher prediction accuracy.
The model achieved an R² score of 0.88, indicating that 88% of the variance in compressive strength is explained by the input features.
The low error rates confirm that the model is well-suited for predicting cement strength with high reliability.

Key Outcomes & Future Scope:
This model provides a data-driven approach for optimizing cement formulations, reducing material costs while maintaining structural strength.

This project successfully demonstrates how machine learning and deep learning techniques can revolutionize material science and construction engineering. 🚀



## Model Building Teachniques
To build an accurate machine learning model for predicting ultimate compressive strength of cement composite mixtures, the following techniques were applied:

1️⃣ Data Preprocessing & Feature Engineering
✅ Handling Missing Values: Checked for null values and used appropriate imputation techniques if needed.
✅ Feature Selection: Identified key independent variables affecting compressive strength.
✅ Feature Scaling (MinMaxScaler):

Applied MinMaxScaler to normalize numerical features.
Ensured all features contributed equally to the learning process.
2️⃣ Model Selection & Implementation
🔹 Linear Regression (Baseline Model)
Implemented Multiple Linear Regression to establish a baseline performance.
Checked the relationship between features and output strength.
🔹 Artificial Neural Network (ANN)
Designed a Deep Learning model with multiple layers.

Architecture:

Input Layer: Accepts cement composition features.
Hidden Layers: Used multiple dense layers with ReLU activation for learning complex patterns.
Output Layer: A single neuron with linear activation to predict compressive strength.
Optimization & Tuning:

Used Adam Optimizer for efficient weight updates.
Mean Squared Error (MSE) as the loss function.
Hyperparameter tuning: Adjusted number of neurons, layers, learning rate, and batch size for optimal performance.
3️⃣ Model Evaluation & Performance Metrics
✅ R² Score (0.88): Measured how well the model explained variance in compressive strength.
✅ Mean Absolute Error (MAE): Evaluated absolute differences between actual and predicted values.
✅ Mean Squared Error (MSE): Penalized larger errors to ensure precise predictions.

### Conclusion
In this study, the variation of contents of GP and waste slags (SS and 
GGBS) on UCS and electrical resistivity were explored by conducting 
laboratory experiments. In addition, we proposed a nature-inspired RF 
algorithm to model the UCS and electrical resistivity. The simulation 
results agree well with the experimental results. The obtained results are 
concluded as follows.  

1) The UCS of ECCC significantly decreases with only a small portion of 
GP content due to its lubrication effect. A 4% GP ratio is recom
mended to achieve satisfactory mechanical and electrical properties 
of ECCC.
 
3) The UCS is slightly enhanced when s/b ratio is less than 20% after 
which, UCS decreases dramatically. However, the extent of influence 
is different. Compared to GGBS, SS demonstrates a less negative 
impact on uniaxial compressive stress during the hydration period.
 
5) The conductivity is improved by the addition of GP, SS and GGBS. Of 
the three conductive fillers, the GP demonstrates the highest 
enhancement effect. Compared with GGBS, the SS is superior 
because it has a less negative influence on UCS and higher 
conductivity.
  
7) The hybrid BAS-RF model has high accuracy in predicting UCS and 
resistivity with correlation coefficients of 0.986 and 0.98, respec
tively on the 30% test sets. The results of the parametric study 
simulated by the RF model agree well with the experimental results.
 
9) Variable importance measured by the RF shows that age has the most 
significant influence on the UCS, while GP is the most important 
variable for resistivity. This result also confirms the experimental 
result.
