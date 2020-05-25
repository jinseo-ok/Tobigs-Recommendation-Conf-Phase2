# Tobigs 추천 컨퍼런스 Phase 2 

> Trip Advisor 데이터 분석 및 CF 기반 모델링

## Getting Started

#### Convention

1. 파일 맨 앞에 자신 이름의 이니셜 대문자 2글자로 한다. (ex SY_TA_crawler.py, SY_TA_EDA.ipynb)
2. 나중에 같이 써야할 파일의 경우) 이니셜 없는 파일을 협의를 통해 만들도록 한다.
3. 우선은 기능별 폴더에 각자 다른 파일을 생성하되 원할 경우 폴더를 더 만들도록 한다. 
4. 모든 데이터는 최상위 data 폴더에 넣고 같은 경로에서 불려올 수 있도록 한다.

#### 경로 설정
1. 경로 접근시 통일성을 위해 반드시 `os.path.join("..","..","data","data.json")` 와 같은 path join을 사용한다.
2. 절대 `pd.read_csv("C://tobigs//data")` 이런 코드가 있어선 안된다.




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












