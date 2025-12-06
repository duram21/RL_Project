import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
from connexion_env import ConnexionEnv

# ───────────────────────── 설정값 (하이퍼파라미터 튜닝) ─────────────────────────
SEED = 42
MAX_EPISODES = 10000        
START_CURRICULUM = 2000     # 조금 더 일찍 난이도를 올리기 시작
END_CURRICULUM = 8000       # 8000판까지 천천히

# PPO 강화 설정
LR = 2.5e-4                 # 학습률 미세 조정
GAMMA = 0.99
K_EPOCH = 10                # 데이터 재사용 횟수 증가 (3 -> 10)
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 4096      # 배치 사이즈 키움 (2048 -> 4096)
MINI_BATCH_SIZE = 64        # 미니배치 줄임 (더 자주 업데이트)
ENTROPY_COEF = 0.01         # 탐험 유도

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── ResNet 신경망 (뇌 업그레이드) ─────────────────────────

# 잔차 블록 (ResBlock): 깊게 쌓아도 학습이 잘 되게 함
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual # 핵심: 입력을 다시 더해줌
        return F.relu(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        
        # 입력: 64개 셀 * 16채널 -> 8x8 이미지로 해석
        # 초기 특징 추출
        self.conv_in = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        
        # ResNet 타워 (뇌 용량 증가)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.res4 = ResBlock(64)
        
        # Pooling (8x8 -> 4x4)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 손패 처리 (MLP)
        self.fc_hand = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 결합부 (CNN 64ch * 4 * 4 = 1024 + Hand 128)
        self.fc_shared = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.ReLU()
        )
        
        # Actor & Critic Heads
        self.actor = nn.Linear(512, act_dim)
        self.critic = nn.Linear(512, 1)

        # 가중치 초기화 (학습 안정성)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, obs):
        batch_size = obs.size(0)
        board_flat = obs[:, :1024]
        hand_flat = obs[:, 1024:]
        
        # Board: [Batch, 16, 8, 8]
        board_img = board_flat.view(batch_size, 64, 16).permute(0, 2, 1).contiguous().view(batch_size, 16, 8, 8)
        
        x = F.relu(self.bn_in(self.conv_in(board_img)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        x = self.pool(x) # -> [Batch, 64, 4, 4]
        x = x.view(batch_size, -1) # Flatten -> 1024
        
        h = self.fc_hand(hand_flat)
        
        combined = torch.cat([x, h], dim=1)
        feat = self.fc_shared(combined)
        return feat

    def get_action(self, obs, mask, action=None):
        feat = self.forward(obs)
        logits = self.actor(feat)
        
        # 마스킹
        logits = logits.masked_fill(~mask, -1e9)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat)

# ───────────────────────── 학습 루프 ─────────────────────────

def train():
    print(f"Training Start on {DEVICE} (Model: ResNet-4, SmartMask: ON)")
    env = ConnexionEnv(seed=SEED)
    
    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 학습률 스케줄러 (점점 정교하게 학습)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=MAX_EPISODES)
    
    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=50); diff_q = deque(maxlen=50)
    best_avg_diff = -999.0

    for ep in range(1, MAX_EPISODES + 1):
        # 난이도 조절
        if ep < START_CURRICULUM: curr_prob = 0.0
        elif ep < END_CURRICULUM:
            progress = (ep - START_CURRICULUM) / (END_CURRICULUM - START_CURRICULUM)
            curr_prob = 0.8 * progress # 최대 80%
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
                # unsqueeze로 배치 차원 추가
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
        
        # 에피소드 끝날 때마다 스케줄러 스텝
        scheduler.step()

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
    # 데이터 변환
    obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32).to(DEVICE)
    act = torch.tensor(np.array(memory["act"]), dtype=torch.long).to(DEVICE)
    logp_old = torch.tensor(np.array(memory["logp"]), dtype=torch.float32).to(DEVICE)
    rew = torch.tensor(np.array(memory["rew"]), dtype=torch.float32).to(DEVICE)
    mask = torch.tensor(np.array(memory["mask"]), dtype=torch.bool).to(DEVICE)
    vals = torch.tensor(np.array(memory["val"]), dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(np.array(memory["done"]), dtype=torch.bool).to(DEVICE)
    
    # GAE 계산
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
    
    # Advantage Normalization (학습 안정성 필수)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO Epoch Update
    dataset_size = len(obs)
    indices = np.arange(dataset_size)
    
    for _ in range(K_EPOCH):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, MINI_BATCH_SIZE):
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
            # Gradient Clipping (폭주 방지)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

# 평가 함수 (동일)
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