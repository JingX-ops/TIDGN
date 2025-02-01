# TIDGN
TIDGN is a transfer learning-based model for predicting intrinsically disordered protein interaction sites using invariant geometric dynamic graph network.

![TIDGN 示例图片](https://github.com/JingX-ops/TIDGN/blob/main/Image/TIDGN.png?raw=true)
# **Requirements**
Python 3.9.16  
numpy 1.25.0  
pandas 1.3.5  
scikit_learn 1.2.2  
torch 1.12.1  
torch_geometric 2.3.0  
torch_geometric_temporal 0.54.0  
tqdm 4.65.0  
# **Reproduce our work**
We provide the complete dataset, processed features, trained models, and corresponding code for the four IDP interaction prediction tasks, allowing reproduction of our work.
- For the FUS RGG homotypic interaction prediction task:
```bash
Python fus_homo_pred.py
```
- For the FUS RGG heterotypic interaction prediction task:
```bash
Python fus_hete_pred.py
```
- For the LAF-1 RGG homotypic interaction prediction task:
```bash
Python laf_homo_pred.py
```
- For the LAF-1 RGG heterotypic interaction prediction task:
```bash
Python laf_hete_pred.py
```

# **Train a new model**
To retrain TIDGN on the four IDP interaction datasets, you can run the following code:
- For the FUS RGG homotypic interaction prediction task:  
```bash
Python fus_homo_train.py
```
- For the FUS RGG heterotypic interaction prediction task:  
```bash
Python fus_hete_train.py
```
- For the LAF-1 RGG homotypic interaction prediction task:  
```bash
Python laf_homo_train.py
```  
- For the LAF-1 RGG heterotypic interaction prediction task:  
```bash
Python laf_hete_train.py
```  



