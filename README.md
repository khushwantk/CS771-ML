## Domain adaptation & Feature Fusion

This repository showcases solutions to complex machine learning challenges involving **continual adaptation** and **heterogeneous data integration**. The projects address real-world scenarios where models must evolve with shifting data distributions and leverage diverse feature representations for robust performance.

---

## 🚀 Projects

### 1. **Multi-Modal Feature Integration for Binary Classification**  
**Problem**: Training a unified model on **3 heterogeneous feature representations** (emoticons, deep embeddings, text sequences) derived from the same raw data.  
**Complexity**:  
- **Feature Heterogeneity**: Combining categorical, high-dimensional (13x786), and sequential (length-50 strings) data.  
- **Data Efficiency**: Optimizing model performance with minimal training data (20%–100% subsets).  
- **Parameter Constraints**: Limited to 10,000 trainable parameters to enforce simplicity and generalizability.  

---
### 2. **Continual Domain Adaptation for Image Classification**
**Problem**: Incrementally updating a model across 20 datasets with **dynamically shifting input distributions** (CIFAR-10 subsets).  
**Complexity**:  
- **Catastrophic Forgetting**: Updating models on new domains without degrading performance on previous tasks.  
- **Distribution Shifts**: Datasets 11–20 introduce unique input distributions with partial similarity to earlier domains.  
- **Label Scarcity**: Only the first dataset is labeled; subsequent updates rely on pseudo-labels.  

---
