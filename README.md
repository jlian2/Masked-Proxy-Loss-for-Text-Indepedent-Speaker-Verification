#### Implementation for Masked Proxy Loss for Text-Independent Speaker Recognition(Interspeech 2021):


https://www.isca-speech.org/archive/interspeech_2021/lian21_interspeech.html

```
@inproceedings{lian21_interspeech,
  author={Jiachen Lian and Aiswarya Vinod Kumar and Hira Dhamyal and Bhiksha Raj and Rita Singh},
  title={{Masked Proxy Loss for Text-Independent Speaker Verification}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4638--4642},
  doi={10.21437/Interspeech.2021-2190}
}
```


The baseline model and code framework is based on:

```
@article{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  journal={arXiv preprint arXiv:2003.11982},
  year={2020}
}
```

#### Training

```
Train on mp_balance(Recommend):

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mp_balance --optimizer sgd --save_path res_model/test_mmp_balance --batch_size 200 --lr 0.2 --max_frames 350 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1

Train on mmp_balance:

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mmp_balance2 --optimizer sgd --save_path res_model/test_mmp_balance2 --batch_size 200 --lr 0.2 --max_frames 350 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1

Train on mp:

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mp --optimizer sgd --save_path res_model/test_mmp_balance --batch_size 200 --lr 0.2 --max_frames 200 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1

Train on mmp:

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mmp --optimizer sgd --save_path res_model/test_mmp_balance --batch_size 200 --lr 0.2 --max_frames 200 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1

```
#### Eval: 

```
One of good models on 4s segment:

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mmp_balance --optimizer sgd --save_path res_model/test_mmp_balance --batch_size 200 --lr 0.2 --max_frames 400 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1 --eval --initial_model voxceleb_pretrained.model
```

#### One example of Pretrained Models (More will be added later)


https://drive.google.com/file/d/1GlktCa1CsZdB9VAN5kK22st2hOMDDAit/view?usp=sharing

