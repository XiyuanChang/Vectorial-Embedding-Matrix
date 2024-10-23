# Vectorial-Embedding-Matrix
Objective: to explore a framework which transfers multi-relational data with missing values and categorical features to 
low-dimensional vector representation in machine learning applications, which ensures the accuracy of data;

· Transformed multi-relational data into low-dimensional matrix through knowledge graph embedding representation 
methods, which can reveal inferences and relationships that are not directly addressed by a single relational table;

· Conducted evaluations on binary and multi-classification datasets using: a baseline logistic regression with one-hot 
encoding, XGBoost model with one-hot encoding and fixed-length knowledge embeddings, and a neural network 
BiLSTM-Attention model with variable-length knowledge embeddings;

· F1-score evaluation revealed that embedding-based models outperformed those using one-hot encoding by 7% to 46%, 
highlighting embeddings' superior capacity for information conveyance, confirming the effectiveness of embedding 
vectorsin enhancing data representation and relational understanding, thereby improving classification task performance.

![image](https://github.com/user-attachments/assets/80fde99b-f9f8-4cd9-a6eb-a4f51c726129)


![Bilstm attention model concept architecture](https://github.com/user-attachments/assets/8292fcfb-fdad-4f52-a711-667a5d6e20e8)


# Citation
```yaml
@inproceedings{tissot2018hextrato,
  title={HEXTRATO: Using ontology-based constraints to improve accuracy on learning domain-specific entity and relationship embedding representation for knowledge resolution},
  author={Tissot, Hegler},
  booktitle={IC3K 2018-Proceedings of the 10th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management},
  volume={10},
  pages={72--81},
  year={2018},
  organization={SciTePress}
}

@article{li2019attention,
  title={An attention-based deep learning model for clinical named entity recognition of Chinese electronic medical records},
  author={Li, Luqi and Zhao, Jie and Hou, Li and Zhai, Yunkai and Shi, Jinming and Cui, Fangfang},
  journal={BMC Medical Informatics and Decision Making},
  volume={19},
  pages={1--11},
  year={2019},
  publisher={Springer}
}
```


