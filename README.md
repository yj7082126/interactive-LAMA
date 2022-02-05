# Large Mask Inpaiting - Interactive ver.

[LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/saic-mdal/lama) 의 웹용 애플리케이션 개발 repo 입니다.

작성자 : 권용재

목적 : 웹툰 이미지 정제를 위한 image inpainting (캐릭터, 효과음, 말풍선 등의 제거로 순수한 배경용 일러스트를 추출)
* StyleGAN, CycleGAN 등의 GAN-based model들에 기존의 배경 데이터셋을 사용시, 상기된 요소들로 인한 image artifact들이 발생함
* 다소의 정보 손실을 감수하고 실제 이미지 도메인에서 좋은 성과를 보인 LaMa를 사용한 image inpainting을 적용

LaMa 관련 링크 : 

[Project page](https://saic-mdal.github.io/lama-project/) [arXiv](https://arxiv.org/abs/2109.07161) [Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf) [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt) [Casual GAN Papers Summary](https://www.casualganpapers.com/large-masks-fourier-convolutions-inpainting/LaMa-explained.html)