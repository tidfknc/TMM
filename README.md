The code for paper: 《Structure Aware Multi-Graph Network for Multi-Modal Emotion Recognition in Conversations》


## Requirements

- numpy 1.19.2
- torchmetrics 0.8.2
- Python 3.6.13
- PyTorch 1.6.0+cu10.2


The code has been tested on Ubuntu 16.04 using a single A100 GPU.
<br>

## Explanations
- ./ckpts: the saved checkpoints on multiple runs 
- ./data: preprocessed datasets and features
- ./logs: training log


## Run Steps

```bash
# For IEMOCAP:
bash iemocap_tav_evaluation.sh
# For MELD:
bash meld_ta_evaluation.sh
bash meld_tav_evaluation.sh
```
