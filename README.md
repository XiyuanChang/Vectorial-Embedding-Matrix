# Vectorial-Embedding-Matrix
Objective: Explored relational embedding techniques to preserve semantic context in sparse tabular data, enhancing both classification accuracy and data explainability without imputing missing values or altering observed data;
```yaml
· Transformed structured tabular data into multi-relational triples to express each feature-value pair as a semantically interpretable relation, then applied knowledge graph embedding techniques to learn compact vector representations for downstream classification;

· Conducted evaluations on binary and multi-classification datasets using: a baseline logistic regression with one-hot 
encoding, XGBoost model with one-hot encoding and fixed-length knowledge embeddings, and a neural network 
BiLSTM-Attention model with variable-length knowledge embeddings;

· F1-score evaluation revealed that embedding-based models outperformed those using one-hot encoding by 7% to 46%, 
highlighting embeddings' superior capacity for information conveyance, confirming the effectiveness of embedding 
vectorsin enhancing data representation and relational understanding, thereby improving classification task performance.
```
![image](https://github.com/user-attachments/assets/80fde99b-f9f8-4cd9-a6eb-a4f51c726129)
![image](https://github.com/user-attachments/assets/f53f15b1-e057-4cf4-8830-1583440fa9c3)


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


