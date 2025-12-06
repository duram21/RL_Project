import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import copy
from connexion_env_7 import ConnexionEnv

# ───────────────────────── 설정값 ─────────────────────────
SEED = 42
MAX_EPISODES = 20000        # 총 5만 판
SELF_PLAY_START = 3000      # 3000판부터 Self-Play 시작

# 업데이트 설정
UPDATE_OPPONENT_INT = 1000  # 1000판마다
UPDATE_OFFSET = 50          # 50판 늦게 (예: 3050, 4050...)

LR = 1e-4
GAMMA = 0.99
K_EPOCH = 10
EPS_CLIP = 0.1
UPDATE_TIMESTEP = 4096      
MINI_BATCH_SIZE = 128       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── ResNet-5 구조 (동일) ─────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x): return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])
        self.fc_hand = nn.Sequential(nn.Linear(160, 128), nn.ReLU())
        self.actor_conv = nn.Conv2d(128, 32, 1); self.actor_bn = nn.BatchNorm2d(32)
        self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)
        self.critic_conv = nn.Conv2d(128, 8, 1); self.critic_bn = nn.BatchNorm2d(8)
        self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256); self.critic_fc2 = nn.Linear(256, 1)

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
        pol = F.relu(self.actor_bn(self.actor_conv(x))).view(b, -1)
        pol = torch.cat([pol, h], dim=1)
        logits = self.actor_fc(pol)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None: action = dist.sample()
        val = F.relu(self.critic_bn(self.critic_conv(x))).view(b, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        value = self.critic_fc2(val)
        return action, dist.log_prob(action), dist.entropy(), value

# ───────────────────────── 학습 루프 (Best Model 기반 업데이트) ─────────────────────────

def train():
    print(f"Training Start on {DEVICE} (Mode: Self-Play vs Best Model)")
    env = ConnexionEnv(seed=SEED)
    
    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=1e-6)
    
    opponent_model = None
    
    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=100); diff_q = deque(maxlen=100)
    
    global_best_diff = -999.0      
    current_opp_best_diff = 0.0    

    for ep in range(1, MAX_EPISODES + 1):
        # 1. 상대방 및 난이도 결정
        if ep >= SELF_PLAY_START:
            # 33% 확률로 Self-Play (Best Model), 33% 슈퍼 봇(1.0), 34% 일반 봇(0.5)
            roll = random.random()
            if roll < 0.33 and opponent_model is not None:
                current_opp = opponent_model
                env.set_greedy_prob(0.0) 
            elif roll < 0.66:
                current_opp = None
                env.set_greedy_prob(1.0) # 슈퍼 봇 (풀파워)
            else:
                current_opp = None
                env.set_greedy_prob(0.5) # 일반 봇
        else:
            current_opp = None
            env.set_greedy_prob(0.0) 

        # 2. 상대방 업데이트 로직 (Best Model 로드)
        if ep > SELF_PLAY_START and (ep - UPDATE_OFFSET) % UPDATE_OPPONENT_INT == 0:
            print(f"\n🔄 [Update Opponent] 상대를 '역대 최고 모델(Best)'로 교체합니다! (Ep {ep})")
            
            # 현재까지의 Best Model을 로드해서 상대로 설정
            if os.path.exists("best_model.pt"):
                opponent_model = copy.deepcopy(model) # 구조 복사
                opponent_model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
                opponent_model.eval()
                print("   -> 🏆 Best Model 로드 완료 (가장 강력한 적 등장)")
                
                # 백업도 남겨둠
                import shutil
                shutil.copy("best_model.pt", f"best_model_stage_{ep}.pt")
            else:
                # 혹시 파일이 없으면 현재 모델 사용
                opponent_model = copy.deepcopy(model)
                opponent_model.eval()
                print("   -> (Best 파일 없음) 현재 모델로 대체")

            # 새로운 강적을 만났으니 기준 점수 리셋
            current_opp_best_diff = 0.0 
            print("   -> 평가 기준 리셋 완료\n")

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
            
            next_obs, rew, done, _, info = env.step(a, opp_model=current_opp, device=DEVICE)
            
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
        
        # 3. 로그 및 저장
        if ep % 50 == 0:
            avg_score = sum(score_q) / len(score_q)
            avg_diff = sum(diff_q) / len(diff_q) if diff_q else 0
            
            opp_name = "Self-Play(Best)" if current_opp else f"Greedy({env.greedy_prob})"
            print(f"[Ep {ep:05d}] Diff: {avg_diff:.1f} | Score: {avg_score:.2f} | Opp: {opp_name}")
            
            save_trigger = False
            
            if current_opp is not None:
                # Self-Play: 0점 이상 & 신기록일 때 저장
                if avg_diff > current_opp_best_diff and avg_diff > 0:
                    current_opp_best_diff = avg_diff
                    save_trigger = True
                    print(f"  --> 👑 Beat Best Self! (+{avg_diff:.1f}) Saved.")
            else:
                # Greedy: 전체 신기록일 때 저장
                if avg_diff > global_best_diff:
                    global_best_diff = avg_diff
                    save_trigger = True
                    print(f"  --> 📈 Global Best vs Greedy! ({global_best_diff:.1f}) Saved.")
            
            if save_trigger:
                torch.save(model.state_dict(), "best_model.pt")

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