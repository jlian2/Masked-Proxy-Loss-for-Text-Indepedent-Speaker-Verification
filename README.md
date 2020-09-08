#### Mask Proxy Loss for Text-Independent Speaker Recognition


https://drive.google.com/file/d/1XQ2vLhQWnRXUfiS-JR_g9m_taNVS11fR/view?usp=sharing


#### Dependencies
```
pip install -r requirements.txt
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
SOTA model on 4s segment: (EER = 2.0308%)

python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc mmp_balance --optimizer sgd --save_path res_model/test_mmp_balance --batch_size 200 --lr 0.2 --max_frames 400 --train_list /home/ubuntu/voxceleb/data/train_list.txt --test_list /home/ubuntu/voxceleb/data/veri_list.txt --train_path /home/ubuntu/voxceleb/data/voxceleb2 --test_path /home/ubuntu/voxceleb/data/voxceleb1 --eval --initial_model voxceleb_pretrained.model
```

#### Pretrained Model


https://drive.google.com/file/d/1GlktCa1CsZdB9VAN5kK22st2hOMDDAit/view?usp=sharing


#### Data

VoxCeleb: (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

train_list.txt identities from VoxCeleb2 dev set(5994 classes)   
veri_list.txt contains evaluation pairs from VoxCeleb1 test set


#### Citation

The baseline model is based on:

```
@article{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  journal={arXiv preprint arXiv:2003.11982},
  year={2020}
}
```

#### License
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
