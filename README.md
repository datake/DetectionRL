# Implementation for "A Distance-based Anomaly Detection Framework for Deep Reinforcement Learning" (TMLR 2024)

This repository contains the Python implementation of the paper "[A Distance-based Anomaly Detection Framework for Deep Reinforcement Learning](https://openreview.net/forum?id=TNKhDBV6PA)" (**TMLR 2024**). We are now constantly updating our code.....


## Part 1: Offline Detection

- Step 1: use the half clean data to do mean and variance estimation for MD, and calibrate quantile threshold for MD+C
- Step 2: use the clean data to generate noisy data
- Step 3: evaluate the detection on both the clean and noisy data

```
python main_detect.py --game cartpole --pca 1 --feature 50 --conformal 1
```  

### Evaluation Model:

The model is trained and saved in ` ./offline/model/` 

### Hyper-parameters

- pca: 1 if we use PCA in the last year to do the dimension reduction
- feature: 50 the reduced feature dimension
- conformal: 1 if we use conformal method to determined the thresholding
- Flag_Random = True we evaluate the random noises
- Flag_ADV = True we evaluate the adversarial noises
- Flag_OOD = True we evaluate the OOD noises

The current implementation is based on two classical control and six Atari games. The code for carla will be updated shortly...


## Part 2: Online Detection (to be updated) 





## Contact

Please contact hongmin2@ualberta.ca or ksun6@ualberta.ca if you have any questions.

## Reference
Please cite our paper if you use our implementation in your research:
```
@article{zhang2021distance,
  title={A Distance-based Anomaly Detection Framework for Deep Reinforcement Learning},
  author={Zhang, Hongming and Sun, Ke and Xu, Bo and Kong, Linglong and M{\"u}ller, Martin},
  journal={Transactions on Machine Learning Research (TMLR) },
  year={2024}
}
