import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
from connexion_env import ConnexionEnv

# ───────────────────────── 하이퍼파라미터 (ResNet 전용 튜닝) ─────────────────────────
SEED = 42
MAX_EPISODES = 20000        # ResNet은 오래 배워야 합니다 (최소 2만)
START_CURRICULUM = 3000     
END_CURRICULUM = 15000      

LR = 1e-4                   # 학습률을 조금 낮춰서 정밀하게 학습
GAMMA = 0.99
K_EPOCH = 10                # 반복 학습 횟수 증가
EPS_CLIP = 0.1              # 클리핑을 좁게 해서 안정성 확보 (0.2 -> 0.1)
UPDATE_TIMESTEP = 4096      # 배치 사이즈 2배 증가 (안정적 그래디언트)
MINI_BATCH_SIZE = 128       # 미니배치 증가
ENTROPY_COEF = 0.02         # 탐험을 더 많이 하도록 유도 (0.01 -> 0.02)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── AlphaZero 스타일 ResNet ─────────────────────────

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
        
        # 1. 입력 처리 (16채널 -> 128채널 확장)
        self.conv_in = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)
        
        # 2. ResNet Tower (깊이 쌓기)
        # AlphaZero는 19~39개 블록을 쓰지만, 우리는 4~6개면 충분함
        self.res_blocks = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128) # 5 Layer Deep
        )
        
        # 3. 손패 처리 (MLP) - 별도 처리 후 합치기
        self.fc_hand = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU()
        )
        
        # 4-1. Policy Head (Actor) - 어디에 둘까?
        self.actor_conv = nn.Conv2d(128, 32, kernel_size=1) # 1x1 Conv로 차원 축소
        self.actor_bn = nn.BatchNorm2d(32)
        self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim) # Board(32ch) + Hand(128)
        
        # 4-2. Value Head (Critic) - 지금 유리한가?
        self.critic_conv = nn.Conv2d(128, 8, kernel_size=1) # 더 강하게 축소
        self.critic_bn = nn.BatchNorm2d(8)
        self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward_shared(self, obs):
        batch_size = obs.size(0)
        board_flat = obs[:, :1024]
        hand_flat = obs[:, 1024:]
        
        # Board: [Batch, 16, 8, 8]
        board_img = board_flat.view(batch_size, 64, 16).permute(0, 2, 1).contiguous().view(batch_size, 16, 8, 8)
        
        # Common ResNet Body
        x = F.relu(self.bn_in(self.conv_in(board_img)))
        x = self.res_blocks(x)
        
        # Hand Feature
        h = self.fc_hand(hand_flat)
        
        return x, h

    def get_action(self, obs, mask, action=None):
        x, h = self.forward_shared(obs)
        batch_size = obs.size(0)
        
        # --- Actor Path ---
        # Conv 1x1 -> BN -> ReLU -> Flatten -> Concat Hand -> FC
        pol = F.relu(self.actor_bn(self.actor_conv(x)))
        pol = pol.view(batch_size, -1)
        pol = torch.cat([pol, h], dim=1)
        logits = self.actor_fc(pol)
        
        # Masking
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        # --- Critic Path ---
        # Conv 1x1 -> BN -> ReLU -> Flatten -> Concat Hand -> FC -> FC
        val = F.relu(self.critic_bn(self.critic_conv(x)))
        val = val.view(batch_size, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        value = self.critic_fc2(val)
        
        return action, dist.log_prob(action), dist.entropy(), value

    # 수읽기(MCTS)에서 사용할 forward 함수
    def forward(self, obs):
        # MCTS용으로 Actor Logits만 반환하는 약식 함수
        x, h = self.forward_shared(obs)
        batch_size = obs.size(0)
        pol = F.relu(self.actor_bn(self.actor_conv(x)))
        pol = pol.view(batch_size, -1)
        pol = torch.cat([pol, h], dim=1)
        return self.actor_fc(pol)

    # Value만 필요할 때
    def get_value(self, obs):
        x, h = self.forward_shared(obs)
        batch_size = obs.size(0)
        val = F.relu(self.critic_bn(self.critic_conv(x)))
        val = val.view(batch_size, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        return self.critic_fc2(val)

# ───────────────────────── 학습 루프 (동일) ─────────────────────────

def train():
    print(f"Training Start on {DEVICE} (Model: AlphaZero Style ResNet)")
    env = ConnexionEnv(seed=SEED)
    
    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # L2 Regularization 추가
    
    # 학습률 스케줄러 (Cosine Annealing: 천천히 줄어들게)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=1e-6)
    
    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=50); diff_q = deque(maxlen=50)
    best_avg_diff = -999.0

    for ep in range(1, MAX_EPISODES + 1):
        # 난이도 조절
        if ep < START_CURRICULUM: curr_prob = 0.0
        elif ep < END_CURRICULUM:
            progress = (ep - START_CURRICULUM) / (END_CURRICULUM - START_CURRICULUM)
            curr_prob = 0.8 * progress
        else: curr_prob = 0.8
        
        env.set_greedy_prob(curr_prob)
        obs, _ = env.reset()
        done = False
        ep_rew = 0
        
        while not done:
            mask = env.get_smart_action_mask()
            
            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
            mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE)
            
            with torch.no_grad():
                action, log_prob, _, val = model.get_action(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            
            a = action.item()
            next_obs, rew, done, _, info = env.step(a)
            
            memory["obs"].append(obs); memory["act"].append(a); memory["logp"].append(log_prob.item())
            memory["rew"].append(rew); memory["mask"].append(mask); memory["done"].append(done); memory["val"].append(val.item())
            
            obs = next_obs
            ep_rew += rew
            
            if len(memory["obs"]) >= UPDATE_TIMESTEP:
                update_ppo(model, optimizer, memory)
                for k in memory: memory[k] = []
                scheduler.step() # 업데이트 단위로 스케줄링

        score_q.append(ep_rew)
        if "diff" in info: diff_q.append(info["diff"])
        
        if ep % 20 == 0:
            avg_score = sum(score_q) / len(score_q)
            avg_diff = sum(diff_q) / len(diff_q) if diff_q else 0
            cur_lr = optimizer.param_groups[0]['lr']
            
            print(f"[Ep {ep:05d}] Diff: {avg_diff:.1f} | Score: {avg_score:.2f} | Prob: {curr_prob:.2f} | LR: {cur_lr:.6f}")
            
            if curr_prob > 0.1 and avg_diff > best_avg_diff:
                best_avg_diff = avg_diff
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  --> New Best Model Saved! (Diff: {best_avg_diff:.1f})")

    torch.save(model.state_dict(), "final_model.pt")
    print("Training Finished.")
    evaluate_policy(model, env, episodes=100)

def update_ppo(model, optimizer, memory):
    # (기존과 동일하지만, MiniBatch 사이즈가 커졌음)
    obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32).to(DEVICE)
    act = torch.tensor(np.array(memory["act"]), dtype=torch.long).to(DEVICE)
    logp_old = torch.tensor(np.array(memory["logp"]), dtype=torch.float32).to(DEVICE)
    rew = torch.tensor(np.array(memory["rew"]), dtype=torch.float32).to(DEVICE)
    mask = torch.tensor(np.array(memory["mask"]), dtype=torch.bool).to(DEVICE)
    vals = torch.tensor(np.array(memory["val"]), dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(np.array(memory["done"]), dtype=torch.bool).to(DEVICE)
    
    vals_next = torch.cat([vals[1:], torch.tensor([0.0], device=DEVICE)])
    deltas = rew + GAMMA * vals_next * (~dones) - vals
    deltas = deltas.cpu().numpy()
    adv_list = []; gae = 0
    for delta, is_done in zip(reversed(deltas), reversed(dones.cpu().numpy())):
        if is_done: gae = 0
        gae = delta + GAMMA * 0.95 * gae
        adv_list.insert(0, gae)
    advantages = torch.tensor(adv_list, dtype=torch.float32).to(DEVICE)
    returns = advantages + vals
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    indices = np.arange(len(obs))
    for _ in range(K_EPOCH):
        np.random.shuffle(indices)
        for start in range(0, len(obs), MINI_BATCH_SIZE):
            idx = indices[start : start + MINI_BATCH_SIZE]
            
            _, log_prob, entropy, val_pred = model.get_action(obs[idx], mask[idx], action=act[idx])
            
            ratio = torch.exp(log_prob - logp_old[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages[idx]
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(val_pred.squeeze(), returns[idx])
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + ENTROPY_COEF * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

def evaluate_policy(model, env, episodes=200):
    print(f"\n=== Final Evaluation (vs Random) ===")
    env.set_greedy_prob(0.0)
    wins = 0; losses = 0; draws = 0
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_smart_action_mask()
            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t, mask_t)
            obs, _, done, _, info = env.step(action.item())
        diff = info.get("diff", 0)
        if diff > 0: wins += 1
        elif diff < 0: losses += 1
        else: draws += 1
    print(f"Win Rate: {(wins/episodes*100):.1f}%")

if __name__ == "__main__":
    train()