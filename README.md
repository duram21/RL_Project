# 🎮 Connexion PPO Agent Project

이 프로젝트는 **PPO(Proximal Policy Optimization)** 알고리즘을 사용하여 보드게임 **Connexion**을 플레이하는 강화학습 에이전트를 학습시키고, 다양한 상대를 대상으로 성능을 평가하는 시스템입니다.

Connexion 문제 : https://nypc.github.io/2025-codebattle/finals_1

게임 시뮬레이터 : https://dh6l222gjj2c6.cloudfront.net/1p



---

## 결과 예시 


![SampleWin](https://github.com/user-attachments/assets/3a9e460d-00f5-4810-bde6-b6e795273556)




## 🛠 실험 환경 (Environment)

본 프로젝트는 아래 환경에서 테스트 및 검증되었습니다.

| Software | Version |
| :--- | :--- |
| **Python** | `3.11.9` |
| **PyTorch** | `2.5.1+cu121` |
| **NumPy** | `2.3.3` |
| **Cuda** | `12.8` |
---

## 🚀 모델 학습 (Training)

### 1. 사전 준비
학습을 수행하기 위해 아래 두 파일이 **같은 폴더** 내에 위치해야 합니다.
* `connexion_env.py` : Connexion 게임 환경 데이터
* `ppo_connexion.py` : PPO 학습 알고리즘 수행 파일

### 2. 학습 실행
터미널에서 아래 명령어를 입력하면 `ppo_connexion.py`에 설정된 `MAX_EPISODES` 만큼 학습이 진행됩니다.

```bash
python ppo_connexion.py
```

### 3. 학습 결과 (Output)
학습이 종료되면 폴더에 두 가지 모델 파일이 저장됩니다.

- **`best_model.pt`** : 학습 과정 중 가장 좋은 성능을 보인 모델
- **`final_model.pt`** : 학습이 모두 끝난 시점의 최종 모델

> **📂 필수 후속 작업**
> 테스트를 진행하기 위해, 생성된 `.pt` 파일을 `test/sample` 폴더로 반드시 이동시켜 주어야 합니다.


## 🛠 Connexion Testing CLI 실행 준비

**1. 입력 데이터 설정**
- **`input.txt`** : 현재 진행할 게임의 패(Tile) 입력 순서가 정의된 파일입니다.

**2. 필수 파일 구성 (`sample` 폴더)**
테스트 실행을 위해 `sample` 폴더 내에 아래 파일들이 반드시 존재해야 합니다.

- `best_model.pt` (학습된 모델)
- `connexion_env.py`
- `player_ppo.py`
- `random-code.py`
- `sample-code.py`
- `strong.py`

| 파일명 | 설명 |
| :--- | :--- |
| **`random-code.py`** | 무작위로 수를 두는 Agent |
| **`sample-code.py`** | 현재 가장 높은 점수를 얻는 수를 선택하는 Agent |
| **`strong.py`** | Greedy + Heuristic 알고리즘을 적용한 Agent |



### 테스트 실행 방법

1. `test` 폴더로 이동한다.
2. 원하는 상대에 맞게 `config.ini` 파일을 수정한다.  
   - 예:  
     ```ini
     EXEC2 = python sample-code.py
     ```  
   - `{ }` 안에는 아래 중 하나를 입력할 수 있다.
     - `sample-code.py` : 현재 턴에서 **가장 높은 점수**를 얻는 수를 두는 에이전트  
     - `random-code.py` : **무작위 수**를 두는 에이전트  
     - `strong.py` : **greedy + heuristic**을 적용한 강한 에이전트  
3. 아래 명령어로 배틀을 실행한다.
   ```bash
   python run_battle.py
   ```
4. `"🎮 몇 판을 진행하시겠습니까? (숫자 입력): "` 라는 문구가 뜨면 원하는 판 수를 입력한다.  
5. 각 판마다 게임 결과를 터미널에서 확인할 수 있다.  
6. 각 게임의 로그는 `logs` 폴더에 `battle.log_xxx.txt` 형식으로 저장되며,  
   모든 게임 결과는 마지막에 `battle_summary.txt` 파일로 저장된다.
 
