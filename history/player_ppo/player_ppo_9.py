#!/usr/bin/env python3
import sys
import os
import random
import copy

# ★ 초기 실행 속도를 위해 라이브러리 지연 로딩

def main():
    model = None
    device = None
    torch = None
    np = None
    
    # ------------------------------------------------------------------
    # 1. 기본 설정 (Env에서 가져옴)
    # ------------------------------------------------------------------
    try:
        from connexion_env_9 import ConnexionEnv, ALL_CELLS, CELL_INDEX, Tile, Color, Symbol, tile_to_id, id_to_tile
    except ImportError:
        sys.exit(1)

    # 문자열 변환 헬퍼
    def get_tile_str(tid):
        t = id_to_tile(tid)
        return t.color.value + t.symbol.value

    def parse_tile(s):
        if s == "X0": return None
        try:
            c_idx = ["R", "G", "B", "Y"].index(s[0])
            s_idx = ["1", "2", "3", "4"].index(s[1])
            return c_idx * 4 + s_idx
        except: return None

    # 상태 변수
    my_hand = []
    opp_hand = []
    board_state = [-1] * 64
    env_sim = None

    # ------------------------------------------------------------------
    # 2. 하이브리드 결정 로직
    # ------------------------------------------------------------------
    def get_hybrid_action(model, env, obs, mask, device, torch, np):
        # 1. PPO 추천
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            ppo_action = torch.argmax(logits, dim=1).item()

        # 2. Greedy 추천
        greedy_action, greedy_score = find_best_greedy_move(env)
        
        # 3. PPO 점수
        ppo_score = simulate_score(env, ppo_action)
        
        # 4. 결정 (Greedy가 10점 이상 높으면 Greedy 선택)
        if greedy_score > ppo_score + 10.0:
            return greedy_action
        else:
            return ppo_action

    def find_best_greedy_move(env):
        best_val = -9999
        best_action = -1
        
        for slot in range(len(env.agent_hand)):
            for c_idx in range(64):
                if env.board[c_idx] == -1:
                    action = slot * 64 + c_idx
                    val = simulate_score(env, action)
                    if val > best_val:
                        best_val = val
                        best_action = action
        
        if best_action == -1: return 0, -9999
        return best_action, best_val

    def simulate_score(env, action):
        backup_board = list(env.board)
        backup_hand = list(env.agent_hand)
        backup_filled = env.filled
        
        slot = action // 64; c_idx = action % 64
        if slot >= len(env.agent_hand) or env.board[c_idx] != -1: return -9999
        
        tid = env.agent_hand.pop(slot)
        env.board[c_idx] = tid
        env.filled += 1
        
        fs, ss, diff = env._compute_scores()
        
        env.board = backup_board
        env.agent_hand = backup_hand
        env.filled = backup_filled
        return diff

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
                    # 🔥 [디버깅] 모델 로드 시작 알림
                    sys.stderr.write("DEBUG: Starting Model Load...\n")
                    
                    try:
                        import torch as t
                        import torch.nn as nn
                        import torch.nn.functional as F
                        import numpy as n
                        
                        torch = t
                        np = n
                        torch.set_num_threads(1)
                        device = torch.device("cpu") # 실행은 CPU
                        
                        # ───────────────────────── ResNet-5 구조 ─────────────────────────
                        class ResBlock(nn.Module):
                            def __init__(self, channels):
                                super().__init__()
                                self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                                self.bn1 = nn.BatchNorm2d(channels)
                                self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                                
                                self.bn2 = nn.BatchNorm2d(channels)
                            def forward(self, x):
                                return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

                        class ActorCritic(nn.Module):
                            def __init__(self, obs_dim, act_dim):
                                super(ActorCritic, self).__init__()
                                
                                # Env 설정에 따라 입력 차원 자동 계산 (1184 = 64*16 + 160)
                                self.board_feat_dim = 64 * 16
                                self.hand_feat_dim = obs_dim - self.board_feat_dim

                                self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
                                self.bn_in = nn.BatchNorm2d(128)
                                self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])

                                self.fc_hand = nn.Sequential(nn.Linear(self.hand_feat_dim, 128), nn.ReLU())

                                # Shared Feature Dimension
                                self.actor_in_dim = 32 * 8 * 8 + 128

                                # Actor Head
                                self.actor_conv = nn.Conv2d(128, 32, 1)
                                self.actor_bn = nn.BatchNorm2d(32)
                                self.actor_fc = nn.Linear(self.actor_in_dim, act_dim)

                                # 🔥 [추가됨] Delta Head (이게 없어서 에러남)
                                self.delta_head = nn.Sequential(
                                    nn.Linear(self.actor_in_dim, self.actor_in_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.actor_in_dim, act_dim),
                                    nn.Tanh(),
                                )

                                # Critic Head
                                self.critic_conv = nn.Conv2d(128, 8, 1)
                                self.critic_bn = nn.BatchNorm2d(8)
                                self.critic_in_dim = 8 * 8 * 8 + 128
                                self.critic_fc1 = nn.Linear(self.critic_in_dim, 256)
                                self.critic_fc2 = nn.Linear(256, 1)

                            def forward_shared(self, obs):
                                b = obs.size(0)
                                board_flat = obs[:, :self.board_feat_dim]
                                hand_flat = obs[:, self.board_feat_dim:]
                                
                                board = board_flat.view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8)
                                x = self.res_blocks(F.relu(self.bn_in(self.conv_in(board))))
                                h = self.fc_hand(hand_flat)
                                return x, h

                            def forward(self, obs):
                                # Policy, Value, Delta 모두 반환 (학습 코드와 동일 구조)
                                x, h = self.forward_shared(obs)
                                b = obs.size(0)
                                
                                # Policy
                                pol_feat = F.relu(self.actor_bn(self.actor_conv(x))).view(b, -1)
                                pol_feat = torch.cat([pol_feat, h], dim=1)
                                logits = self.actor_fc(pol_feat)
                                
                                # Delta (사용은 안 해도 로딩을 위해 계산)
                                delta = self.delta_head(pol_feat)
                                
                                # Value
                                val_feat = F.relu(self.critic_bn(self.critic_conv(x))).view(b, -1)
                                val_feat = torch.cat([val_feat, h], dim=1)
                                val = self.critic_fc2(F.relu(self.critic_fc1(val_feat))).squeeze(-1)
                                
                                return logits, val, delta

                            # Value만 필요할 때 쓰는 헬퍼 함수
                            def get_value(self, obs):
                                _, val, _ = self.forward(obs)
                                return val

                        model = ActorCritic(1184, 320).to(device)
                        
                        # 모델 로드
                        model_path = "best_model_duram2.pt" # (현재 학습중인 파일명 확인 필요)
                        if not os.path.exists(model_path): model_path = "best_model.pt"
                        if not os.path.exists(model_path): model_path = "final_model.pt"
                        
                        if os.path.exists(model_path):
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            # 🔥 [디버깅] 모델 로드 성공 알림
                            sys.stderr.write(f"DEBUG: Model Loaded Successfully from {model_path}\n")
                        else:
                            sys.stderr.write("DEBUG: WARNING! No model file found. Running with random weights.\n")
                        
                        model.eval()
                        env_sim = ConnexionEnv()

                        # 워밍업
                        with torch.no_grad():
                            dummy_obs = torch.zeros(1, 1184).to(device)
                            _ = model.forward_shared(dummy_obs)

                    except Exception as e:
                        sys.stderr.write(f"DEBUG: Error during model loading: {e}\n")

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]
                opp_hand = [parse_tile(t) for t in parts[6:11]]
                board_state = [-1] * 64
                if env_sim:
                    env_sim.reset()
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:]
                    env_sim.board = board_state[:]

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
                    
                    action = get_hybrid_action(model, env_sim, obs, mask, device, torch, np)
                else:
                    action = 0

                slot, c_idx = action // 64, action % 64
                
                if board_state[c_idx] != -1:
                    for i in range(64):
                        if board_state[i] == -1: c_idx = i; slot = 0; break
                if slot >= len(my_hand): slot = 0
                
                print(f"PUT {str(ALL_CELLS[c_idx])} {get_tile_str(my_hand[slot])}")
                sys.stdout.flush()
                
                board_state[c_idx] = my_hand.pop(slot)

            elif cmd == "GET":
                new_t = parse_tile(parts[1])
                if new_t is not None: my_hand.append(new_t)

            elif cmd == "OPP":
                c_idx = CELL_INDEX.get(parts[1])
                t1_id = parse_tile(parts[2]); t2_id = parse_tile(parts[3])
                if c_idx is not None:
                    board_state[c_idx] = t1_id
                    if t1_id in opp_hand: opp_hand.remove(t1_id)
                    elif opp_hand: opp_hand.pop(0)
                    if t2_id is not None: opp_hand.append(t2_id)

            elif cmd == "FINISH":
                break

        except Exception:
            break

if __name__ == "__main__":
    main()