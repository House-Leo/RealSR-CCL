# RWSR-CCL 

Criteria Comparative Learning for Real-scene Image Super-Resolution 
Yukai Shi, Hao Li, Sen Zhang, Zhijing Yang, Xiao Wang

*Guangdong University of Technology, The University of Sydney, Peng Cheng Laboratory*

(*Official PyTorch Implementation*)

## Update - July, 2022
- Release testing code and pre-trained model for NTIRE 2020.

## Introduction

Real-scene image super-resolution aims to restore real-world low-resolution images into their high-quality versions. A typical RealSR framework usually includes the optimization of multiple criteria which are designed for different image properties, by making the implicit assumption that the ground-truth images can provide a good trade-off between different criteria. However, this assumption could be easily violated in practice due to the inherent contrastive relationship between different image properties. Contrastive learning (CL) provides a promising recipe to relieve this problem by learning discriminative features using the triplet contrastive losses. Though CL has achieved significant success in many computer vision tasks, it is non-trivial to introduce CL to RealSR due to the difficulty in defining valid positive image pairs in this case. Inspired by the observation that the contrastive relationship could also exist between the criteria, in this work, we propose a novel training paradigm for RealSR, named Criteria Comparative Learning (Cria-CL), by developing contrastive losses defined on criteria instead of image patches. In addition, a spatial projector is proposed to obtain a good view for Cria-CL in RealSR. Our experiments demonstrate that compared with the typical weighted regression strategy, our method achieves a significant improvement under similar parameter settings. 


## Citation:
If you find this work useful for your research, please cite:

```
@artical{shi2022realsr,
  title={Criteria Comparative Learning for Real-scene Image Super-Resolution},
  author={Shi, Yukai and Li, Hao and Zhang, Sen and Yang, Zhijing and Wang, Xiao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022}
}
```

## Download Link
- Pre-trained model [Google drive](https://drive.google.com/file/d/1busHlGDlf-YOY5tQSEy5uOhzsl9deJn-/view?usp=sharing)
- [NTIRE 2020](https://competitions.codalab.org/competitions/22221)
- [RealSR](https://github.com/csjcai/RealSR)
- [CameraSR](https://github.com/ngchc/CameraSR)

## How to use the code during test phase.

1. `git clone https://github.com/house-leo/RealSR-CCL`
2. Put your model script under the `models` folder.
3. Put your pretrained model under the `model_zoo` folder.
4. Modify `model_path` in `test_demo.py`. Modify the imported models.
5. `python test_demo.py`
6. Get metrics results by running `python caculate_metric.py`. 

## Contact:
Please contact me if there is any question (Hao Li: haoli@njust.edu.cn) (Yukai Shi: ykshi@gdut.edu.cn).