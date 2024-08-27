# Universal-Overwrite-Attack

Source code of KDD'25 submission (resubmission) "Universal Overwrite Attack against DNN-based Image Watermarking Models".


## Environment
- Python 3.8.10
- numpy==1.23.5
- scikit-learn==1.3.0
- torch=1.12.1+cu113
- torchmetrics==0.6.0
- torchvision==0.13.1+cu113
- tensorflow==2.13.0
- tensorboard=2.13.0
- Pillow=9.5.0
- huggingface-hub==0.16.4
- compressai==1.2.4
- pytorch-fid==0.3.0

You could use the following instruction to install all the requirements:
```
pip install -r requirements.txt
```

## Run UOA
As an example, to run the UOA attack against HiDDeN on COCO dataset, you could use the following command:
```
python run_uoa.py --watermarking-model hidden --wm-ckpt-file-path /path/to/model/checkpoint --data-root-dir /path/to/image/folder --dataset coco --message-length 30 --img-size 128
```

## Run AUOA
As an example, to run the AUOA attack against HiDDeN on COCO dataset, you could use the following command:
```
python run_auoa.py --k 100 --watermarking-model hidden --wm-ckpt-file-path /path/to/model/checkpoint --data-root-dir /path/to/image/folder --dataset coco --message-length 30 --img-size 128
```


