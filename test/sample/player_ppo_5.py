#!/usr/bin/env python3
import sys
import os
import random

# 초기 실행 속도 최적화 (Lazy Import)
def main():
    model = None
    device = None
    torch = None
    np = None
    
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

    my_hand = []
    board_state = [-1] * 64
    env_sim = None

    # Hybrid 결정 로직
    def get_hybrid_action(model, env, obs, mask, device, torch, np):
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

        greedy_action, greedy_score = find_best_greedy_move(env)
        ppo_score = simulate_score(env, ppo_action)
        
        # Greedy가 5점 이상 유리하면 Greedy 선택, 아니면 PPO의 큰 그림 선택
        if greedy_score > ppo_score + 5.0: return greedy_action
        else: return ppo_action

    def find_best_greedy_move(env):
        best_val = -9999
        best_action = -1
        for slot in range(len(env.agent_hand)):
            for c_idx in range(64):
                if env.board[c_idx] == -1:
                    action = slot * 64 + c_idx
                    val = simulate_score(env, action)
                    if val > best_val: best_val = val; best_action = action
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

    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            parts = line.strip().split()
            if not parts: continue
            cmd = parts[0]

            if cmd == "READY":
                print("OK"); sys.stdout.flush()
                
                if model is None:
                    import torch as t
                    import torch.nn as nn
                    import torch.nn.functional as F
                    import numpy as n
                    from connexion_env_11 import ConnexionEnv 
                    torch = t; np = n; torch.set_num_threads(1)
                    
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
                            super().__init__()
                            self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
                            self.bn_in = nn.BatchNorm2d(128)
                            self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])
                            self.fc_hand = nn.Sequential(nn.Linear(80, 128), nn.ReLU())
                            self.actor_conv = nn.Conv2d(128, 32, 1)
                            self.actor_bn = nn.BatchNorm2d(32)
                            self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)
                            self.critic_conv = nn.Conv2d(128, 8, 1)
                            self.critic_bn = nn.BatchNorm2d(8)
                            self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
                            self.critic_fc2 = nn.Linear(256, 1)
                        def forward_shared(self, obs):
                            b = obs.size(0)
                            board = obs[:, :1024].view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8)
                            hand = obs[:, 1024:]
                            x = self.res_blocks(F.relu(self.bn_in(self.conv_in(board))))
                            h = self.fc_hand(hand)
                            return x, h

                    device = torch.device("cpu")
                    model = ActorCritic(1104, 320).to(device)
                    model_path = "final_model_duram.pt"
                    if not os.path.exists(model_path): model_path = "final_model_5.pt"
                    if os.path.exists(model_path):
                        try: model.load_state_dict(torch.load(model_path, map_location=device))
                        except: pass
                    model.eval()
                    env_sim = ConnexionEnv()
                    with torch.no_grad():
                        dummy_obs = torch.zeros(1, 1104).to(device)
                        _ = model.forward_shared(dummy_obs)

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]
                board_state = [-1] * 64
                if env_sim: env_sim.reset(); env_sim.agent_hand = my_hand[:]; env_sim.board = board_state[:]

            elif cmd == "TIME":
                obs = np.zeros(1104, dtype=np.float32)
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    action = get_hybrid_action(model, env_sim, obs, mask, device, torch, np)
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
                c_idx = cell_to_idx.get(parts[1])
                if c_idx is not None: board_state[c_idx] = parse_tile(parts[2])

            elif cmd == "FINISH": break
        except Exception: break

if __name__ == "__main__":
    main()