#!/usr/bin/env python3
import sys
import os
import random
import time  # 시간 측정을 위해 추가

# ★ 초기 실행 속도를 위해 라이브러리 지연 로딩

def main():
    model = None
    device = None
    torch = None
    np = None
    
    # ------------------------------------------------------------------
    # 1. 기본 설정
    # ------------------------------------------------------------------
    colors = ['R', 'G', 'B', 'Y']
    symbols = ['1', '2', '3', '4']
    
    valid_cells = []
    forbidden = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}
    for c in ['a','b','c','d','e','f']:
        for r in ['1','2','3','4','5','6']:
            for s in ['-', '+']:
                if (c+r+s) not in forbidden: valid_cells.append(c+r+s)
    
    cell_to_idx = {s: i for i, s in enumerate(valid_cells)}

    def parse_tile(s):
        if s == "X0": return None
        try: return colors.index(s[0]) * 4 + symbols.index(s[1])
        except: return None

    def get_tile_str(tid):
        return colors[tid//4] + symbols[tid%4]

    # 상태 변수
    my_hand = []
    opp_hand = []
    board_state = [-1] * 64
    env_sim = None

    # ------------------------------------------------------------------
    # 2. 배치 처리 최적화 로직 (속도 10배 향상)
    # ------------------------------------------------------------------
    def get_batch_value_action(model, env, obs, mask, device, torch, np):
        start_time = time.time()
        
        K = 10 # 후보 수
        
        # 1. Policy로 후보 추천
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(1, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=1)
            
            top_values, top_indices = torch.topk(probs, min(K, int(mask.sum())))
            candidates = top_indices[0].cpu().numpy()

        # 2. 배치 데이터 준비 (Batch Preparation)
        # 루프 돌면서 모델을 부르지 않고, 데이터를 모아서 한 방에 처리
        
        next_obs_list = []
        valid_candidates = []
        
        # 상태 백업
        bak_board = list(env.board)
        bak_hand = list(env.agent_hand)
        bak_filled = env.filled
        
        for action in candidates:
            slot = action // 64; c_idx = action % 64
            if slot >= len(env.agent_hand) or env.board[c_idx] != -1: continue

            # 가상 착수
            tid = env.agent_hand.pop(slot)
            env.board[c_idx] = tid
            env.filled += 1
            
            # 다음 상태 관측값 저장
            next_obs_list.append(env._get_obs())
            valid_candidates.append(action)
            
            # 복구
            env.board = list(bak_board)
            env.agent_hand = list(bak_hand)
            env.filled = bak_filled
            
            # [Time Cut] 시간 없으면 중단 (0.2초 넘으면)
            if time.time() - start_time > 0.2:
                break
        
        if not valid_candidates: return candidates[0]

        # 3. 일괄 추론 (Batch Inference) 🔥 여기가 핵심 속도 향상 포인트
        next_obs_tensor = torch.tensor(np.array(next_obs_list)).float().to(device)
        
        with torch.no_grad():
            # 한 번의 호출로 모든 후보의 가치 계산
            values = model.get_value(next_obs_tensor).cpu().numpy().flatten()

        # 4. 최적의 수 선택
        best_score = -99999.0
        best_action = valid_candidates[0]
        
        # 방어 로직 계산 (이건 파이썬 연산이라 빠름)
        empty_cells = [i for i, x in enumerate(env.board) if x == -1]
        
        for i, action in enumerate(valid_candidates):
            val_pred = values[i]
            
            # 방어 점수 (간략화: 상위 2개만 샘플링하거나 생략 가능)
            # 시간이 없으면 방어 계산 스킵
            if time.time() - start_time > 0.4: # 0.4초 넘으면 방어 계산 포기
                max_opp_gain = 0
            else:
                max_opp_gain = 0
                if env.opp_hand:
                    # 상대 패 중 2개만 샘플링해서 검사 (속도 타협)
                    for opp_tid in env.opp_hand[:2]: 
                        for opp_c in empty_cells:
                            if env.board[opp_c] == -1:
                                g = env._local_score(opp_c, opp_tid, is_first=False)
                                if g > max_opp_gain: max_opp_gain = g
            
            # 최종 점수
            final_score = (val_pred * 20.0) - (max_opp_gain * 1.0)
            
            if final_score > best_score:
                best_score = final_score
                best_action = action

        return best_action

    # ------------------------------------------------------------------
    # 3. 메인 루프
    # ------------------------------------------------------------------
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            parts = line.strip().split()
            if not parts: continue
            cmd = parts[0]

            if cmd == "READY":
                print("OK")
                sys.stdout.flush()
                
                if model is None:
                    import torch as t; import torch.nn as nn; import torch.nn.functional as F; import numpy as n; from connexion_env import ConnexionEnv 
                    torch = t; np = n; torch.set_num_threads(1); device = torch.device("cpu")
                    
                    # ResNet-5 (학습 코드와 동일)
                    class ResBlock(nn.Module):
                        def __init__(self, channels): super().__init__(); self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(channels); self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(channels)
                        def forward(self, x): return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
                    class ActorCritic(nn.Module):
                        def __init__(self, obs_dim, act_dim):
                            super(ActorCritic, self).__init__(); self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False); self.bn_in = nn.BatchNorm2d(128); self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)]); self.fc_hand = nn.Sequential(nn.Linear(160, 128), nn.ReLU()); self.actor_conv = nn.Conv2d(128, 32, 1); self.actor_bn = nn.BatchNorm2d(32); self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim); self.critic_conv = nn.Conv2d(128, 8, 1); self.critic_bn = nn.BatchNorm2d(8); self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256); self.critic_fc2 = nn.Linear(256, 1)
                        def forward_shared(self, obs): b = obs.size(0); board = obs[:, :1024].view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8); hand = obs[:, 1024:]; x = self.res_blocks(F.relu(self.bn_in(self.conv_in(board)))); h = self.fc_hand(hand); return x, h
                        def get_value(self, obs): x, h = self.forward_shared(obs); b=x.size(0); val = F.relu(self.critic_bn(self.critic_conv(x))).view(b, -1); val = torch.cat([val, h], dim=1); val = F.relu(self.critic_fc1(val)); return self.critic_fc2(val)

                    model = ActorCritic(1184, 320).to(device)
                    model_path = "best_model.pt"; 
                    if not os.path.exists(model_path): model_path = "final_model.pt"
                    if os.path.exists(model_path):
                        try: model.load_state_dict(torch.load(model_path, map_location=device))
                        except: pass
                    model.eval(); env_sim = ConnexionEnv()

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]; opp_hand = [parse_tile(t) for t in parts[6:11]]; board_state = [-1] * 64
                if env_sim: env_sim.reset(); env_sim.agent_hand = my_hand[:]; env_sim.opp_hand = opp_hand[:]; env_sim.board = board_state[:]

            elif cmd == "TIME":
                obs = np.zeros(1184, dtype=np.float32)
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                for i in range(5):
                    if i < len(opp_hand): obs[1104 + i*16 + opp_hand[i]] = 1.0
                
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:]
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    
                    # 🔥 [배치 처리] 속도 최적화 버전 호출
                    action = get_batch_value_action(model, env_sim, obs, mask, device, torch, np)
                else: action = 0

                slot, c_idx = action // 64, action % 64
                if board_state[c_idx] != -1:
                    for i in range(64):
                        if board_state[i] == -1: c_idx = i; slot = 0; break
                if slot >= len(my_hand): slot = 0
                
                print(f"PUT {valid_cells[c_idx]} {get_tile_str(my_hand[slot])}")
                sys.stdout.flush()
                board_state[c_idx] = my_hand.pop(slot)

            elif cmd == "GET":
                new_t = parse_tile(parts[1])
                if new_t is not None: my_hand.append(new_t)
            elif cmd == "OPP":
                c_idx = cell_to_idx.get(parts[1]); t1_id = parse_tile(parts[2]); t2_id = parse_tile(parts[3])
                if c_idx is not None:
                    board_state[c_idx] = t1_id
                    if t1_id in opp_hand: opp_hand.remove(t1_id)
                    elif opp_hand: opp_hand.pop(0)
                    if t2_id is not None: opp_hand.append(t2_id)
            elif cmd == "FINISH": break
        except Exception: break

if __name__ == "__main__":
    main()