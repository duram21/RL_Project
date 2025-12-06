# ppo_connexion.py 7 기준으로 gpt가 refactoring
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import copy
from connexion_env_8 import ConnexionEnv

# ───────────────────────── 설정값 ─────────────────────────
SEED = 21
MAX_EPISODES = 250000
SELF_PLAY_START = 100000        # 3000판부터 self-play / 강한 상대 등장
HEURISTIC_START_EP = 180000     # 8000판 이후에는 휴리스틱 엔진도 opponent pool에 투입
HEURISTIC_PROB = 0.4          # 해당 구간에서 휴리스틱과 싸울 확률

# opponent 업데이트 설정
UPDATE_OPPONENT_INT = 1000    # 1000판마다
UPDATE_OFFSET = 50            # 3050, 4050 ... 처럼 약간 늦게 교체

LR = 1e-4
GAMMA = 0.99
K_EPOCH = 10
EPS_CLIP = 0.1
UPDATE_TIMESTEP = 4096
MINI_BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        if action is None:
            action = dist.sample()
        val = F.relu(self.critic_bn(self.critic_conv(x))).view(b, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        value = self.critic_fc2(val)
        return action, dist.log_prob(action), dist.entropy(), value


# ───────────────────────── PPO 업데이트 ─────────────────────────
def update_ppo(model, optimizer, memory):
    obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32).to(DEVICE)
    act = torch.tensor(np.array(memory["act"]), dtype=torch.long).to(DEVICE)
    logp_old = torch.tensor(np.array(memory["logp"]), dtype=torch.float32).to(DEVICE)
    rew = torch.tensor(np.array(memory["rew"]), dtype=torch.float32).to(DEVICE)
    mask = torch.tensor(np.array(memory["mask"]), dtype=torch.bool).to(DEVICE)
    vals = torch.tensor(np.array(memory["val"]), dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(np.array(memory["done"]), dtype=torch.bool).to(DEVICE)

    # GAE
    vals_next = torch.cat([vals[1:], torch.tensor([0.0], device=DEVICE)])
    deltas = rew + GAMMA * vals_next * (~dones) - vals
    deltas = deltas.cpu().numpy()
    adv_list = []
    gae = 0.0
    for delta, is_done in zip(reversed(deltas), reversed(dones.cpu().numpy())):
        if is_done:
            gae = 0.0
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
            if len(idx) == 0:
                continue

            _, log_prob, entropy, val_pred = model.get_action(obs[idx], mask[idx], action=act[idx])
            ratio = torch.exp(log_prob - logp_old[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages[idx]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(val_pred.squeeze(), returns[idx])
            entropy_loss = -entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + 0.02 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


# ───────────────────────── 학습 루프 ─────────────────────────
def train():
    print(f"Training Start on {DEVICE} (Mode: Self-Play + Heuristic Opponent)")
    env = ConnexionEnv(seed=SEED)

    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=1e-6)

    opponent_model = None

    memory = {"obs": [], "act": [], "logp": [], "rew": [], "mask": [], "done": [], "val": []}
    score_q = deque(maxlen=100); diff_q = deque(maxlen=100)

    global_best_diff = -999.0      # greedy/heuristic 상대로의 최고 diff
    current_opp_best_diff = 0.0    # self-play vs best 기준
    # heuristic 전용 베스트 따로 보고 싶으면 변수 하나 더 만들면 됨

    for ep in range(1, MAX_EPISODES + 1):
        # ── 1. 이번 에피소드 상대 종류 선정 ──
        # opponent_kind: "random", "greedy_0.5", "greedy_1.0", "selfplay_best", "heuristic"
        env.set_heuristic_mode(False)
        opponent_kind = "random"

        if ep >= SELF_PLAY_START:
            # 먼저 "휴리스틱 vs" 할지 결정
            if ep >= HEURISTIC_START_EP and random.random() < HEURISTIC_PROB:
                # 휴리스틱 엔진을 상대
                current_opp = None
                env.set_greedy_prob(0.0)
                env.set_heuristic_mode(True, is_first_for_heuristic=False)  # 휴리스틱을 후공 관점으로 사용
                opponent_kind = "heuristic"
            else:
                # 기존 pool: self-play(best) vs greedy bots
                roll = random.random()
                if roll < 0.33 and opponent_model is not None:
                    current_opp = opponent_model
                    env.set_greedy_prob(0.0)
                    opponent_kind = "selfplay_best"
                elif roll < 0.66:
                    current_opp = None
                    env.set_greedy_prob(1.0)
                    opponent_kind = "greedy_1.0"
                else:
                    current_opp = None
                    env.set_greedy_prob(0.5)
                    opponent_kind = "greedy_0.5"
        else:
            current_opp = None
            env.set_greedy_prob(0.0)
            opponent_kind = "random"

        # ── 2. 상대 업데이트(역대 최고 모델로 갱신) ──
        if ep > SELF_PLAY_START and (ep - UPDATE_OFFSET) % UPDATE_OPPONENT_INT == 0:
            print(f"\n🔄 [Update Opponent] 상대를 '역대 최고 PPO 모델'로 교체합니다! (Ep {ep})")
            if os.path.exists("best_model.pt"):
                opponent_model = copy.deepcopy(model)
                opponent_model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
                opponent_model.eval()
                print("   -> 🏆 best_model.pt 로드 완료")
                import shutil
                shutil.copy("best_model.pt", f"best_model_stage_{ep}.pt")
            else:
                opponent_model = copy.deepcopy(model)
                opponent_model.eval()
                print("   -> best_model.pt 없음. 현재 모델로 대체")
            current_opp_best_diff = 0.0
            print("   -> self-play 평가 기준 리셋\n")

        # ── 3. 에피소드 플레이 ──
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        info = {}

        while not done:
            mask = env.get_smart_action_mask()
            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, val = model.get_action(obs_t, mask_t)
            a = action.item()

            next_obs, rew, done, _, info = env.step(a, opp_model=current_opp, device=DEVICE)

            memory["obs"].append(obs)
            memory["act"].append(a)
            memory["logp"].append(log_prob.item())
            memory["rew"].append(rew)
            memory["mask"].append(mask)
            memory["done"].append(done)
            memory["val"].append(val.item())

            obs = next_obs
            ep_rew += rew

            if len(memory["obs"]) >= UPDATE_TIMESTEP:
                update_ppo(model, optimizer, memory)
                for k in memory:
                    memory[k] = []
                scheduler.step()

        score_q.append(ep_rew)
        if "diff" in info:
            diff_q.append(info["diff"])

        # 3. 로그 및 저장 (개선된 버전)
        if ep % 50 == 0:
            avg_score = sum(score_q) / len(score_q)
            avg_diff = sum(diff_q) / len(diff_q) if diff_q else 0
            
            opp_type = opponent_kind  # "heuristic", "selfplay_best" 등
            print(f"[Ep {ep:05d}] Diff: {avg_diff:.1f} | Score: {avg_score:.2f} | Opp: {opp_type}")
            
            save_trigger = False
            
            # Case A: 휴리스틱(최강자) 상대일 때
            if opp_type == "heuristic":
                # [핵심] 이기지 못하더라도(-점수), '역대 휴리스틱 상대 최고 기록'이면 저장!
                # (예: -40점에서 -20점으로 줄였으면 발전한 것임)
                if avg_diff > global_best_diff:  # global_best_diff는 초반에 -9999로 초기화됨
                    global_best_diff = avg_diff
                    save_trigger = True
                    print(f"  --> 🛡️ Best Defense vs Heuristic! ({avg_diff:.1f}) Saved.")

            # Case B: Self-Play 상대일 때
            elif opp_type == "selfplay_best":
                # 여기서는 이겨야 의미가 있음 (Diff > 0)
                if avg_diff > current_opp_best_diff and avg_diff > 0:
                    current_opp_best_diff = avg_diff
                    save_trigger = True
                    print(f"  --> 👑 Beat Best Self! (+{avg_diff:.1f}) Saved.")
            
            # Case C: 일반 Greedy 등
            else:
                # 여기서는 그냥 잘하면 저장 (참고용)
                if avg_diff > 50.0 and avg_diff > global_best_diff:
                     # (Global Best를 휴리스틱이랑 공유하면 꼬일 수 있으니 별도 변수 쓰거나 생략)
                     pass 
            
            if save_trigger:
                torch.save(model.state_dict(), "best_model.pt")

    torch.save(model.state_dict(), "final_model.pt")
    print("Training Finished.")


if __name__ == "__main__":
    train()
