# SMRec

This is the code repository for the paper "Significance-aware Medication Recommendation with Medication Representation Learning".

## Environment
```
python==3.9.18
torch==2.1.1
tqdm==4.66.1
dgl==1.1.2.cu118
scikit-learn==1.3.2
```
You can build the conda environment for our experiment using the following command:
```
conda env create -f environment.yml
```



## Prepare the datasets

All same as [SafeDrug](https://github.com/ycq091044/SafeDrug/) or [COGNet](https://github.com/BarryRun/COGNet) except `process.py`.



## Train or Test


You can train or test the model using the following command:
```
python main.py
python main.py --test True
```

## Acknowledgement
If this work is useful in your research, please cite our paper.

```
@inproceedings{li2024mr,
  title={Significance-aware Medication Recommendation with Medication Representation Learning},
  author={Yishuo Li, Zhufeng Shao, Weimin Chen, Shoujin Wang, Yuehan Du, Wenpeng Lu},
  booktitle={Proceedings of the 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD 2024)}
  year={2024},
  organization={IEEE}
}
```
