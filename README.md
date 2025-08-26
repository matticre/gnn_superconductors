# Detection of Quenches in Superconducting Cables using Graph Neural Networks

## 1. Project Overview and Context

This project focuses on the critical task of detecting "quenches"—a phenomenon where superconducting cables lose their superconductivity—by analyzing sequences of thermal imaging data. The system under consideration is a 15x15 grid of magnesium diboride (MgB₂) superconducting cables. These cables operate at a temperature of 20 K and conduct a current of 18 kA, maintaining their superconducting state below a critical temperature of approximately 25 K.

The input data consists of sequences of heat maps, with each sequence containing 24 hourly temperature measurements of the cable grid. Although the cables are individually cooled, there is still a degree of thermal diffusion across the grid. A quench event, triggered by a sudden temperature increase in one or more cables, can propagate through this grid. This project aims to establish a reliable connection between the thermal data and the occurrence of these quench events.

## 2. Task and Objective

The primary goal of this project is to build a model that can accurately distinguish between sequences with at least one quench event (labeled as class 1) and those with no quenches (class 0).

## 3. Models Employed

To address this challenge, this project explores and compares two primary deep learning architectures: a **Graph Attention Network (GNN)** and a **DeepSet** model. The data is structured as a fully connected graph where each of the 24 time steps in a sequence is represented as a node.

Each node in the graph is enriched with the following features:

* The reshaped 15x15 grid of temperature data.
* The specific time step (1-24).
* The maximum, minimum, and average temperature across the grid for that time step.

### 3.1. Graph Attention Network (GNN)

The GNN architecture is the core of this project, chosen for its high expressiveness and its ability to effectively share information between nodes for graph classification. The GNN was implemented with two different message-passing mechanisms to compare their performance:

* **GNN with DeepSet Message Passing:** This version uses a DeepSet architecture to aggregate and update node information, allowing the model to learn from the collective properties of the time steps.
* **GNN with Attention Mechanism:** This more advanced version incorporates an attention mechanism, enabling the model to weigh the importance of different nodes (time steps) when making a prediction. This is particularly useful as the exploratory data analysis revealed that quenches are more likely to occur in the "central" time steps of a sequence.

### 3.2. DeepSet Model

As a baseline for comparison against the more complex GNNs, a simpler DeepSet model was also implemented. This model processes the features of each time step independently and then aggregates them to produce a final prediction. While less expressive than the GNNs, it provides a valuable benchmark for the performance gains achieved through graph-based message passing.

## 4. Results and Analysis

The models were evaluated with a focus on metrics that are robust to the class imbalance present in the dataset (approximately 90% of the data belongs to the "no quench" class). For this binary classification task, the models were evaluated based on **Accuracy, Precision, True Positive Rate (TPR), and True Negative Rate (TNR)**.

* **GNN with Attention Mechanism:** This model emerged as the top performer, achieving a near-perfect **Accuracy of 99.67%** and a **Precision of 100%**. Most impressively, it obtained a **TPR of 96.36%**, indicating its exceptional ability to correctly identify quench events.
* **GNN with DeepSet Message Passing:** This model also performed well, with an **Accuracy of 96.50%** and a high **Precision of 97.22%**. However, its **TPR of 63.64%** was significantly lower than the attention-based model, suggesting it was less effective at identifying all positive cases.
* **DeepSet Model:** The baseline DeepSet model achieved a respectable **Accuracy of 93.00%** but had a lower **Precision (74.07%)** and **TPR (36.36%)**.

**Conclusion:** The results clearly demonstrate the superiority of the GNN with an attention mechanism. The high TPR of this model is the most critical outcome, as failing to detect a quench event can have serious consequences. The perfect precision also means that when the model predicts a quench, it is always correct.