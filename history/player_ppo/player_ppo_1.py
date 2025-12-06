#!/usr/bin/env python3
import sys
import os
import copy
import random

# ★ 무거운 라이브러리는 나중에 로딩
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
    board_state = [-1] * 64
    
    # 시뮬레이터 객체
    env_sim = None

    # ------------------------------------------------------------------
    # 2. 수읽기 (Lookahead Search) 함수
    # ------------------------------------------------------------------
    def find_best_move_with_search(model, env, obs, mask, device, torch, np):
        K = 5
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model.forward(obs_t)
            logits = model.actor(feat)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=1)
            top_values, top_indices = torch.topk(probs, K)
            candidates = top_indices[0].cpu().numpy()

        best_action = candidates[0]
        best_diff = -9999

        # 백업
        backup_board = list(env.board)
        backup_hand = list(env.agent_hand)
        backup_filled = env.filled

        for action in candidates:
            # 상태 복구 (루프 시작마다)
            env.board = list(backup_board)
            env.agent_hand = list(backup_hand)
            env.filled = backup_filled
            
            slot = action // 64
            c_idx = action % 64
            
            # 유효성 1차 체크 (혹시라도 찬 곳이면 패스)
            if env.board[c_idx] != -1: continue
            if slot >= len(env.agent_hand): continue
            
            # 시뮬레이션
            tid = env.agent_hand.pop(slot)
            env.board[c_idx] = tid
            env.filled += 1
            
            fs, ss, diff = env._compute_scores()
            
            if diff > best_diff:
                best_diff = diff
                best_action = action
        
        # 최종 복구
        env.board = list(backup_board)
        env.agent_hand = list(backup_hand)
        env.filled = backup_filled
        
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
                    import torch as t
                    import torch.nn as nn
                    import torch.nn.functional as F
                    import numpy as n
                    from connexion_env import ConnexionEnv 
                    
                    torch = t
                    np = n
                    torch.set_num_threads(1)
                    
                    class ActorCritic(nn.Module):
                        def __init__(self, obs_dim, act_dim):
                            super().__init__()
                            self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
                            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
                            self.pool = nn.MaxPool2d(2, 2)
                            self.fc_hand = nn.Linear(80, 128)
                            self.fc_common = nn.Linear(1024 + 128, 512)
                            self.actor = nn.Linear(512, act_dim)
                            self.critic = nn.Linear(512, 1)

                        def forward(self, obs):
                            b = obs.size(0)
                            b_img = obs[:, :1024].view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8)
                            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(b_img)))))
                            x = x.view(b, -1)
                            h = F.relu(self.fc_hand(obs[:, 1024:]))
                            return F.relu(self.fc_common(torch.cat([x, h], dim=1)))

                    device = torch.device("cpu")
                    model = ActorCritic(1104, 320).to(device)
                    
                    if os.path.exists("final_model_2.pt"):
                        try: model.load_state_dict(torch.load("final_model_2.pt", map_location=device))
                        except: pass
                    model.eval()
                    env_sim = ConnexionEnv()

                    # 워밍업
                    with torch.no_grad():
                        dummy_obs = torch.zeros(1, 1104).to(device)
                        _ = model.forward(dummy_obs)

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]
                board_state = [-1] * 64
                if env_sim:
                    env_sim.reset()
                    env_sim.agent_hand = my_hand[:]
                    env_sim.board = board_state[:]

            elif cmd == "TIME":
                # Obs 생성
                obs = np.zeros(1104, dtype=np.float32)
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                
                # Mask 생성 (Smart Masking)
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    
                    # 수읽기 실행
                    action = find_best_move_with_search(model, env_sim, obs, mask, device, torch, np)
                else:
                    # Fallback
                    mask = np.zeros(320, dtype=bool)
                    for s_idx in range(len(my_hand)):
                        for c_idx in range(64):
                            if board_state[c_idx] == -1: mask[s_idx*64 + c_idx] = True
                    
                    obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
                    mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.forward(obs_t)
                        logits = model.actor(feat)
                        logits = logits.masked_fill(~mask_t, -1e9)
                        action = torch.argmax(logits, dim=1).item()
                
                slot, c_idx = action // 64, action % 64
                
                # 🛡️ [최종 안전장치] 선택한 칸이 비어있는지 확인
                if board_state[c_idx] != -1:
                    # 에러 상황: 이미 찬 곳을 골랐음 -> 강제로 빈칸 찾기
                    found_fallback = False
                    for fallback_c in range(64):
                        if board_state[fallback_c] == -1:
                            c_idx = fallback_c
                            slot = 0 # 손패 첫 번째 그냥 냄
                            found_fallback = True
                            break
                    # 정말 둘 곳이 없으면(보드 꽉 참) 아무거나 출력(어차피 끝남)
                
                if slot >= len(my_hand): slot = 0
                
                print(f"PUT {valid_cells[c_idx]} {get_tile_str(my_hand[slot])}")
                sys.stdout.flush()
                
                board_state[c_idx] = my_hand.pop(slot)

            elif cmd == "GET":
                new_t = parse_tile(parts[1])
                if new_t is not None: my_hand.append(new_t)

            elif cmd == "OPP":
                # OPP p T1 ...
                c_idx = cell_to_idx.get(parts[1])
                if c_idx is not None: board_state[c_idx] = parse_tile(parts[2])

            elif cmd == "FINISH":
                break

        except Exception:
            break

if __name__ == "__main__":
    main()