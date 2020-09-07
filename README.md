# SinGAN-torch
Reproduce SinGAN: Learning a Generative Model from a Single Natural Image

### 수정된 부분
1. Coarsest generator 학습 시 rec_z 한 번만 생성<br> 
(원래는 epoch 마다 새로 생성하더라 github issue 63번에도 올라온 내용인데 저자가 대답을 안했네)
  
2. Generator iteration 마다 fake image 새로 생성

3. Discriminator iteration 마다 random noise 새로 생성 (아직 수정 x) 

### 체크할 사항
- Origin repo에 비해 학습 속도가 얼마나 느려졌는가 -> 위에 2번 때문에 느려졌을 것 
- SN 쓸 생각은 안해봤댄다
- https://github.com/tamarott/SinGAN/issues/59 읽어볼 것