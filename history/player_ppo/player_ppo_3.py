#!/usr/bin/env python3
import sys
import os
import random
import copy

# ★ 초기 실행 속도를 위해 무거운 라이브러리는 나중에 로딩 (Lazy Import)

def main():
    model = None
    device = None
    torch = None
    np = None
    
    # ------------------------------------------------------------------
    # 1. 가벼운 상수/함수 정의
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
    env_sim = None

    # ------------------------------------------------------------------
    # 2. 수읽기 (Lookahead Search) 함수
    # ------------------------------------------------------------------
    def find_best_move_with_search(model, env, obs, mask, device, torch, np):
        K = 5 # 상위 5개 후보에 대해 시뮬레이션
        
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Actor의 Logits만 가져옴
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=1)
            
            # 상위 K개 인덱스 추출
            top_values, top_indices = torch.topk(probs, K)
            candidates = top_indices[0].cpu().numpy()

        best_action = candidates[0]
        best_diff = -9999

        # 현재 상태 백업
        backup_board = list(env.board)
        backup_hand = list(env.agent_hand)
        backup_filled = env.filled

        for action in candidates:
            # 상태 복구
            env.board = list(backup_board)
            env.agent_hand = list(backup_hand)
            env.filled = backup_filled
            
            slot = action // 64
            c_idx = action % 64
            
            # 유효성 체크 (혹시 모를 오류 방지)
            if env.board[c_idx] != -1: continue
            if slot >= len(env.agent_hand): continue
            
            # 시뮬레이션 (내가 둠)
            tid = env.agent_hand.pop(slot)
            env.board[c_idx] = tid
            env.filled += 1
            
            # 점수 계산 (깊이 1: 내가 뒀을 때의 점수 차이)
            # 심화: 여기서 상대방의 최선의 수까지 예측하면 더 좋음 (Minimax)
            fs, ss, diff = env._compute_scores()
            
            # 여기에 Value Network의 값도 더해주면 금상첨화
            # score = diff * 0.7 + value_pred * 0.3
            
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
                # TLE 방지용 OK 먼저 전송
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
                    
                    # ───────────────────────── ResNet 구조 (학습 코드와 동일!) ─────────────────────────
                    class ResBlock(nn.Module):
                        def __init__(self, channels):
                            super().__init__()
                            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
                            self.bn1 = nn.BatchNorm2d(channels)
                            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
                            self.bn2 = nn.BatchNorm2d(channels)

                        def forward(self, x):
                            residual = x
                            out = F.relu(self.bn1(self.conv1(x)))
                            out = self.bn2(self.conv2(out))
                            out += residual
                            return F.relu(out)

                    class ActorCritic(nn.Module):
                        def __init__(self, obs_dim, act_dim):
                            super(ActorCritic, self).__init__()
                            
                            self.conv_in = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1, bias=False)
                            self.bn_in = nn.BatchNorm2d(128)
                            
                            # 5-Layer ResNet
                            self.res_blocks = nn.Sequential(
                                ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128)
                            )
                            
                            self.fc_hand = nn.Sequential(nn.Linear(80, 128), nn.ReLU())
                            
                            # Actor Head
                            self.actor_conv = nn.Conv2d(128, 32, kernel_size=1)
                            self.actor_bn = nn.BatchNorm2d(32)
                            self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)
                            
                            # Critic Head (구조 맞추기용)
                            self.critic_conv = nn.Conv2d(128, 8, kernel_size=1)
                            self.critic_bn = nn.BatchNorm2d(8)
                            self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
                            self.critic_fc2 = nn.Linear(256, 1)

                        def forward_shared(self, obs):
                            b = obs.size(0)
                            board = obs[:, :1024].view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8)
                            hand = obs[:, 1024:]
                            x = F.relu(self.bn_in(self.conv_in(board)))
                            x = self.res_blocks(x)
                            h = self.fc_hand(hand)
                            return x, h

                    device = torch.device("cpu")
                    model = ActorCritic(1104, 320).to(device)
                    
                    # 모델 로드 (없으면 초기화 상태로 진행 - 에러 방지)
                    model_path = "final_model.pt"
                    if not os.path.exists(model_path): model_path = "final_model.pt"
                    
                    if os.path.exists(model_path):
                        try: model.load_state_dict(torch.load(model_path, map_location=device))
                        except: pass 
                    model.eval()
                    env_sim = ConnexionEnv()

                    # 워밍업
                    with torch.no_grad():
                        dummy_obs = torch.zeros(1, 1104).to(device)
                        _ = model.forward_shared(dummy_obs)

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]
                board_state = [-1] * 64
                if env_sim:
                    env_sim.reset()
                    env_sim.agent_hand = my_hand[:]
                    env_sim.board = board_state[:]

            elif cmd == "TIME":
                obs = np.zeros(1104, dtype=np.float32)
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                
                # 행동 결정
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    
                    # 🔥 ResNet + 수읽기 실행
                    action = find_best_move_with_search(model, env_sim, obs, mask, device, torch, np)
                else:
                    # Fallback (안전장치)
                    action = 0

                slot, c_idx = action // 64, action % 64
                
                # 안전장치: 이미 찬 곳이면 빈 곳 찾기
                if board_state[c_idx] != -1:
                    for i in range(64):
                        if board_state[i] == -1:
                            c_idx = i; slot = 0; break
                            
                if slot >= len(my_hand): slot = 0
                
                print(f"PUT {valid_cells[c_idx]} {get_tile_str(my_hand[slot])}")
                sys.stdout.flush()
                
                board_state[c_idx] = my_hand.pop(slot)

            elif cmd == "GET":
                new_t = parse_tile(parts[1])
                if new_t is not None: my_hand.append(new_t)

            elif cmd == "OPP":
                c_idx = cell_to_idx.get(parts[1])
                if c_idx is not None: board_state[c_idx] = parse_tile(parts[2])

            elif cmd == "FINISH":
                break

        except Exception:
            break

if __name__ == "__main__":
    main()