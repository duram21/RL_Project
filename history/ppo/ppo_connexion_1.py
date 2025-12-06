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
MAX_EPISODES = 500000         # 5천 판만 해도 Smart Masking 덕분에 금방 잘함
START_CURRICULUM = 300000     # 1000판부터 바로 난이도 올림
END_CURRICULUM = 500000       # 마지막엔 Greedy 확률 80%까지 도전

LR = 3e-4
GAMMA = 0.99
K_EPOCH = 3
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 2048
MINI_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc_hand = nn.Linear(80, 128)
        self.fc_common = nn.Linear(1024 + 128, 512)
        self.actor = nn.Linear(512, act_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        batch_size = obs.size(0)
        board_flat = obs[:, :1024]
        hand_flat = obs[:, 1024:]
        board_img = board_flat.view(batch_size, 64, 16).permute(0, 2, 1).contiguous().view(batch_size, 16, 8, 8)
        x = F.relu(self.conv1(board_img))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        h = F.relu(self.fc_hand(hand_flat))
        combined = torch.cat([x, h], dim=1)
        feat = F.relu(self.fc_common(combined))
        return feat

    def get_action(self, obs, mask, action=None):
        feat = self.forward(obs)
        logits = self.actor(feat)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat)

def train():
    print(f"Training Start on {DEVICE} (Model: CNN + Smart Masking)")
    env = ConnexionEnv(seed=SEED)
    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=50); diff_q = deque(maxlen=50)
    best_avg_diff = -999.0

    for ep in range(1, MAX_EPISODES + 1):
        # 난이도 조절: 1000판부터 5000판까지 확률 0.0 -> 0.8로 상승
        if ep < START_CURRICULUM: curr_prob = 0.0
        else:
            progress = (ep - START_CURRICULUM) / (END_CURRICULUM - START_CURRICULUM)
            curr_prob = 0.8 * progress # 최대 80%까지
        
        env.set_greedy_prob(curr_prob)
        obs, _ = env.reset()
        done = False
        ep_rew = 0
        
        while not done:
            # 🔥 여기서 Smart Mask를 사용함!
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

        score_q.append(ep_rew)
        if "diff" in info: diff_q.append(info["diff"])
        
        if ep % 20 == 0:
            avg_score = sum(score_q) / len(score_q)
            avg_diff = sum(diff_q) / len(diff_q) if diff_q else 0
            print(f"[Ep {ep:05d}] GreedyProb: {curr_prob:.2f} | Score: {avg_score:.2f} | Diff: {avg_diff:.1f}")
            
            if curr_prob > 0.1 and avg_diff > best_avg_diff:
                best_avg_diff = avg_diff
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  --> New Best Model Saved! (Diff: {best_avg_diff:.1f})")

    torch.save(model.state_dict(), "final_model.pt")
    print("Training Finished.")
    evaluate_policy(model, env, episodes=100)

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
            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val_pred.squeeze(), returns[idx]) - 0.01 * entropy.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

def evaluate_policy(model, env, episodes=200):
    print(f"\n=== Evaluating Policy over {episodes} Episodes ===")
    
    # 🔥 [수정됨] 상대를 'Random(바보)'으로 설정 (Prob = 0.0)
    test_prob = 0.0 
    env.set_greedy_prob(test_prob)
    
    print(f"Opponent Strength (Greedy Prob): {test_prob} (Random Bot)")
    print("--> Expecting High Win Rate...")

    wins = 0; losses = 0; draws = 0
    total_diff = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 테스트니까 Smart Mask 켜서 최선의 수만 두게 함
            mask = env.get_smart_action_mask() 
            
            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                # Argmax: 확률 말고 가장 점수 높은 수 선택 (더 강력함)
                action, _, _, _ = model.get_action(obs_t, mask_t)
            
            obs, _, done, _, info = env.step(action.item())
        
        diff = info.get("diff", 0)
        total_diff += diff
        
        if diff > 0: wins += 1
        elif diff < 0: losses += 1
        else: draws += 1
        
        if (ep + 1) % 50 == 0:
            print(f"  Processed {ep + 1}/{episodes} episodes... (Current Wins: {wins})")

    win_rate = (wins / episodes) * 100
    avg_diff = total_diff / episodes
    
    print("\n" + "="*40)
    print(f"FINAL RESULT (vs Random Bot)")
    print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Score Diff: {avg_diff:.2f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    train()