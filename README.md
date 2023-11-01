# Privacy-Guaranteed Heart Disease Diagnosis via Federated Learning
#### Matthew Cai, Rohil Agarwal, Hansong Zhou, and Xiaonan Zhang

## Abstract
Heart disease is one of the most prevalent diseases in the world, killing the most people in the world annually. Hence, efficiently and effectively diagnosing heart disease is a crucial task. Recent advancements in machine learning (ML) have empowered healthcare providers to enhance diagnostic accuracy; nevertheless, this usually demands data sharing, posing a severe threat to patients’ privacy. Emerging as a potential solution, federated learning (FL) enables the construction of a powerful learning model without sharing raw data, and has been successfully utilized by various applications, including Google G-board and Apple Siri. This project explores the application of FL to heart disease diagnosis.

Utilizing 80% of the Cleveland hospital’s subset of the UCI ML Repository’s Heart Disease dataset, the program split the data to simulate four distinct hospitals collecting information. Then, the program trains a federated Multilayer Perceptron (MLP) to predict heart disease diagnosis. Training was run iteratively for 1000 global epochs and 3 local epochs per global epoch, with testing of the global model against the remaining 20% of the Cleveland dataset occurring once every 10 global epochs, resulting in a final convergent accuracy of 85%, nearing the 88.3% accuracy of the centralized model. The experimental results demonstrate that federated MLPs can effectively diagnose heart disease while maintaining individual data privacy. The implications of this are extensive, illuminating how firms in healthcare and potentially other privacy-sensitive industries like finance can increase diagnosis efficiency and accuracy of various diseases while continuing to ensure the privacy of their clients.

## Technical Details
_Research conducted at Florida State University, Department of Computer Science during the 2023 Young Scholars Program_

**System Design:** 1 server, 4 clients (hospitals) with randomly distributed data.  
**Dataset:** Cleveland Dataset of [Heart Disease - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)  
**Model:** 
&nbsp;Type: Multilayer Perceptron.  
&nbsp;Layers:  
&nbsp;&nbsp;&nbsp;&nbsp;Layer 1: Input, 13 nodes.  
&nbsp;&nbsp;&nbsp;&nbsp;Layer 2: Hidden, 32 nodes. Activation Function, Leaky ReLU.  
&nbsp;&nbsp;&nbsp;&nbsp;Layer 3: Output, 1 node. Activation Function, Sigmoid.  
&nbsp;Loss: Binary Cross Entropy.  
&nbsp;Optimizer: Adam, learning rate of 1e<sup>-3</sup>.  
&nbsp;Epochs: 1000 global epochs. 3 local epochs per global epoch.  

## Results
Federated learning model produces 85% final convergent accuracy, despite non independent and identically distributed data (Non-I.I.D).  
Centralized learning model produces 88.3% final convergent accuracy.  
Federated and centralized learning model demonstrated similar efficiency.  

## Conclusion
Federated learning can efficienty and accurately emable the use of machine learning in privacy sensitive industries such as healthcare.  
Federated learning can train an accurate global model even in the presence of Non-I.I.D. data between clients.  
Therefore, federated learning is a powerful alternative to centralized learning that can promote increased collaboration between hospitals and other firms in privacy-sensitive fields.  
