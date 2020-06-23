# Tobigs 추천 컨퍼런스 Phase 2 

> Trip Advisor 데이터 분석 및 CF 기반 모델링

## Dashboard

|담당자|분류|설명|해당 파일|
|--|--|--|--|
|SY|EDA|Trip Advisor 데이터의 간단한 EDA|[링크](src/data_prep/SY_TA_EDA.ipynb)|
|YN|Model|FM 모델|[링크](src/model/YN_Factorization_Matrix.ipynb)|
|YN|Model|SVD 모델|[링크](src/model/YN_TripAdvisor_SVD_python.ipynb)|
|YN|Model|Neural Collaborative Filtering 모델|[링크](src/model/YN_NCF.ipynb)|
|YN|Model|Simple Algoritm Recommender 모델|[링크](src/model/YN_SAR_based_Recommder.ipynb)|
|YN|Model|Wide and Deep 모델|[링크](src/model/YN_wide_deep.ipynb)|
|YN|Model|Wide and Deep 모델|[링크](src/model/YN_wide_deep_all.ipynb)|
|YN|Model|SAR 모델|[링크](src/model/YN_SAR_based_Recommender.ipynb)|
|YN|Model|VAE CF 모델|[링크](src/model/YN_vaecf.ipynb)|
|JB|Model|SAR 모델|[링크](src/model/JB_SAR.ipynb)|
|JB|Model|SAR 모델|[링크](src/model/JB_SAR_v2.ipynb)|
|HJ|Model|autoencoder_collaborate_filtering 모델|[링크](src/model/HJ_autoencoder_collaborate_filtering.ipynb)|
|HJ|Model|VAE CF 모델|[링크](src/model/HJ_vae_sar_ver2.ipynb)|

## Getting Started

#### Convention

1. 파일 맨 앞에 자신 이름의 이니셜 대문자 2글자로 한다. (ex SY_TA_crawler.py, SY_TA_EDA.ipynb)
2. 나중에 같이 써야할 파일의 경우) 이니셜 없는 파일을 협의를 통해 만들도록 한다.
3. 우선은 기능별 폴더에 각자 다른 파일을 생성하되 원할 경우 폴더를 더 만들도록 한다. 
4. 모든 데이터는 최상위 data 폴더에 넣고 같은 경로에서 불려올 수 있도록 한다.

#### 경로 설정
1. 경로 접근시 통일성을 위해 반드시 `os.path.join("..","..","data","data.json")` 와 같은 path join을 사용한다.
2. 절대 `pd.read_csv("C://tobigs//data")` 이런 코드가 있어선 안된다.

#### 기본 파일 불러오기 

```
import pandas as pd 
import os

df = pd.read_json(os.path.join("..","..","data","TA_User_Reviws_Korea_all.json"))
df.head(5)
```


## How to use github

#### 1. 파일 용량이 큰 거나 (> 100mb) 민감한 정보는 무조건 .gitignore에 추가한다.

#### 2. 다음 과정을 통해 깃 푸쉬를 진행한다.

``` bash
# 용량 큰 파일 다시 꼭 확인 후
git add .
git commit -m "커밋 메시지"
git pull
# pull 을 통해 서버에 있는 파일과 충돌 확인
# 만약 충돌이 있다면 해당 파일 들어가 해결 후 다시 처음부터 진행
git push
```

## Real Model 
YN_model.py

| parameter     |                                                              |
| :--------- | ------------------------------------------------------------ |
| `local_gloabal`    | local을 추천받을지 global 추천받을지 |
| `model`   | `wnd` or `deepFM` &nbsp; 사용할 모델 지정. |
| `path`   | local_df, global_df, vec이 존재하는 경로 |
| `item_id`   | `int` &nbsp; 1차 추천 호텔 리스트 중 사용자가 선택한 호텔 id 
| `top`   | `int` &nbsp; 상위 몇개의 유사 아이템을 추천받을지. default 값은 10이다.  |


| return value|                                                              |
| :---------- | ------------------------------------------------------------ |
| top n 개의 유사 아이템(식당) 이름 및 주소|

### 사용예 
main.ipynb 참조 
```
! python YN_model.py --local_gloabal 'local' --model 'wnd' --path "../realtime_model" --item_id 3477158 --top 10
```







