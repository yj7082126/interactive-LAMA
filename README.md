# Large Mask Inpaiting - Interactive ver.

[LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/saic-mdal/lama) 의 웹용 애플리케이션 개발 repo 입니다.
이미지 인페인팅 기술을 통해 사진/그림 배경에서의 인물이나, 원하지 않은 대상 등을 자유롭고 자연스럽게 지울 수 있습니다.

LaMa 관련 링크 : 
[[Github Page](https://github.com/saic-mdal/lama)] [[Project page](https://saic-mdal.github.io/lama-project/)] [[arXiv](https://arxiv.org/abs/2109.07161)] [[Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)] [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt)] 

## 사용법 (inference) : 
0. pip install -r requirements.txt 로 필요한 패키지들을 다운로드 합니다.
   * inference로는 맨 위의 7개 패키지만 필요합니다.
   * torch 설치가 잘 되지 않을 경우, [Previous Pytorch Versions](https://pytorch.org/get-started/previous-versions/) 웹페이지를 참고하시기 바랍니다.
1. Trained Checkpoint 다운로드
```
pip3 install wldhx.yadisk-direct #다운로드에 필요한 패키지
cd checkpoints
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
```
2. 인페인팅 시키고 싶은 이미지들을 img 폴더 안으로 이동
3. server.py 실행
```
python server.py \
--address=127.0.0.1 # URL 주소 \ 
--port=6006 # URL 포트 \
--imgdir=img/test/ # 인페인팅 시키고 싶은 이미지 폴더의 주소. img 폴더 안에 있어야 함. \
--model_loc=checkpoints/big-lama #모델 checkpoint 폴더 주소. 
--device=cuda:0 #사용하고 싶은 장치 이름
```
4. 주소 [127.0.0.1/6006](http://127.0.0.1:6006)로 이동

시작화면 : 
![before](img/before.png)
* 왼쪽 사진에 마우스를 클릭한 채 움직이는 걸로 가리고 싶은 영역을 색칠할 수 있습니다.
  * (색은 마젠타 색으로 고정시켜주세요)
  * 마젠타 색 옆의 동그라미 아이콘을 선택하는 것으로 브러시 크기를 바꿀 수 있습니다.
* 위의 파란색 화살표들을 클릭하는 것으로 사진을 선택할 수 있습니다.
* 노란색 Inpainting 버튼을 클릭해서 이미지 인페인팅을 실행합니다.

결과 :
![after](img/after.png)
img/webui/[실행 시점의 timestamp 폴더] 에서 인페인팅 결과를 확인할 수 있습니다.

## ToDo List
- [ ] 드래그 & 드롭 이미지 업로드 위젯 만들기
- [ ] drawingboard 업데이트 

## Citations

기본이 되는 코드는 [정식 LAMA 레포](https://github.com/saic-mdal/lama)에서 가져왔습니다.
```
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```

Interactive drawing을 위한 js 프로그램은 [drawingboard.js](https://github.com/Leimi/drawingboard.js/)에서 가져왔습니다.


