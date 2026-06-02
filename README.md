## ALCHEMIST: Hierarchical Generative Inverse Design for Exploratory Materials Discovery 🚀📈
Deep generative models offer a promising route to inverse materials design, yet practical deployment is limited by the difficulty of reconciling compositional novelty with property fidelity and by the scarcity of structured training data. Here we present ALCHEMIST, a closed-loop framework that integrates LLM-assisted literature mining, hierarchical conditional generative modeling, and experimental validation for data-driven alloy discovery. Applied to metallic glasses, ALCHEMIST extracts data from over 200 publications, then generates candidate alloys in two complementary modes: an exploration mode achieving 70% mode coverage with high compositional novelty, and a precision mode producing targeted candidates with thermal-property deviations below 4%. Experimental synthesis of three previously unreported Zr-based metallic glasses confirms both capabilities.

## **Overall Architecture:** 

<p align="center">
  <img src = "images/flow.png" width="700">
  <br/>
  <br/>
</p>


## Installation

For installing, follow these instructions
```
conda create -n alloygan python=3.8
conda activate alloygan
pip install -r requirements.txt
```

## Data preparation 

The collected data in PDF format has been placed in the main directory`./Collected Alloy Dataset.pdf`. The complete Alloy dataset will be released later and made available for download.

 *Training*
 ---
Running training of GAN model, CGAN model, and HCVAE model on alloy dataset:


```
python train.py --model GAN --dataroot datasets/alloys/Alloy_train.csv --epochs 100 --cuda --batch_size 32

python train.py --model CGAN --dataroot datasets/alloys/Alloy_train.csv --epochs 100 --cuda --batch_size 32

python train.py --model HCVAE --dataroot datasets/alloys/Alloy_train.csv --epochs 300 --cuda --batch_size 32
```

Start tensorboard:

```
tensorboard --logdir ./logs/
```

## Results
<p align="center">
  <img src = "images/chem.png" width="700">
  <br/>
  <br/>
  <b> Experiment validation of three metallic glass candidates: HCVAE-generated Zr<sub>55.5</sub>Cu<sub>18.5</sub>Al<sub>13.5</sub>Co<sub>12.4</sub>, and CGAN-generated Zr<sub>62</sub>Cu<sub>29.6</sub>Al<sub>4.2</sub>Ag<sub>2.3</sub>Ni<sub>1.9</sub> and Zr<sub>62.1</sub>Cu<sub>31.4</sub>Al<sub>5.1</sub>Ni<sub>1.4</sub>
 </b>
</p>

## Citation

If you find this code useful in your research, please consider citing:
```
@article{hao2025inverse,
  title={Inverse Materials Design by Large Language Model-Assisted Generative Framework},
  author={Hao, Yun and Fan, Che and Ye, Beilin and Lu, Wenhao and Lu, Zhen and Zhao, Peilin and Gao, Zhifeng and Wu, Qingyao and Liu, Yanhui and Wen, Tongqi},
  journal={arXiv preprint arXiv:2502.18127},
  year={2025}
}

```
