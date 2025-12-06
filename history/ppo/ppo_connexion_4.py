import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
from connexion_env import ConnexionEnv

# ───────────────────────── 설정값 ─────────────────────────
SEED = 42
MAX_EPISODES = 10000        # 1만 판으로 축소
START_CURRICULUM = 2000     # 2천 판부터 난이도 올리기 시작 (더 빨리 시작)
END_CURRICULUM = 8000       # 8천 판에서 최고 난이도 도달 (밀도 높임)

LR = 1e-4                   # ResNet이니까 학습률은 그대로 유지
GAMMA = 0.99
K_EPOCH = 10
EPS_CLIP = 0.1
UPDATE_TIMESTEP = 4096      
MINI_BATCH_SIZE = 128       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── ResNet-5 모델 ─────────────────────────
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

    def get_action(self, obs, mask, action=None):
        x, h = self.forward_shared(obs)
        b = obs.size(0)
        
        # Actor
        pol = F.relu(self.actor_bn(self.actor_conv(x))).view(b, -1)
        pol = torch.cat([pol, h], dim=1)
        logits = self.actor_fc(pol)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None: action = dist.sample()
            
        # Critic
        val = F.relu(self.critic_bn(self.critic_conv(x))).view(b, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        value = self.critic_fc2(val)
        return action, dist.log_prob(action), dist.entropy(), value

def train():
    print(f"Training Start on {DEVICE} (Model: ResNet-5)")
    env = ConnexionEnv(seed=SEED)
    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=1e-6)
    
    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=100); diff_q = deque(maxlen=100)
    best_avg_diff = -999.0

    for ep in range(1, MAX_EPISODES + 1):
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
                scheduler.step()

        score_q.append(ep_rew)
        if "diff" in info: diff_q.append(info["diff"])
        
        if ep % 50 == 0:
            avg_score = sum(score_q) / len(score_q)
            avg_diff = sum(diff_q) / len(diff_q) if diff_q else 0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"[Ep {ep:05d}] Diff: {avg_diff:.1f} | Score: {avg_score:.2f} | Prob: {curr_prob:.2f}")
            
            if curr_prob > 0.1 and avg_diff > best_avg_diff:
                best_avg_diff = avg_diff
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  --> New Best Model Saved! (Diff: {best_avg_diff:.1f})")

    torch.save(model.state_dict(), "final_model.pt")
    print("Training Finished.")

def update_ppo(model, optimizer, memory):
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
            loss = actor_loss + 0.5 * critic_loss + 0.02 * entropy_loss
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()

if __name__ == "__main__":
    train()