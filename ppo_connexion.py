# ppo_connexion_hybrid.py
import os
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from connexion_env import ConnexionEnv

# ─────────────────────────────────────────────────────────────
# 설정값 (self-play + PPO)
# ─────────────────────────────────────────────────────────────

SEED = 21

# 에피소드 기반 self-play
MAX_EPISODES      = 250_000
SELF_PLAY_START   = 3_000     # 여기부터 best self-play 상대 등장
HEURISTIC_START_EP = 8_000    # 여기부터 휴리스틱 엔진도 opponent pool에 투입
HEURISTIC_PROB    = 0.4         # 해당 구간에서 heuristic opponent로 붙을 확률

# opponent 업데이트 설정
UPDATE_OPPONENT_INT = 1000      # best_model로 opponent 갱신 간격
UPDATE_OFFSET       = 50        # 100050, 101050 ... 처럼 약간 늦게 교체

# PPO 하이퍼파라미터
LR              = 1.0e-4
GAMMA           = 0.99
LAMBDA          = 0.95
EPS_CLIP        = 0.15
UPDATE_TIMESTEP = 4096
K_EPOCH         = 4
MINI_BATCH_SIZE = 256
ENT_COEF        = 0.005
VF_COEF         = 0.5
DELTA_LOSS_COEF = 0.1
DELTA_SCALE     = 20.0
MAX_GRAD_NORM   = 0.5

# KL 안정화
KL_TARGET       = 0.03
KL_STOP_FACTOR  = 4.0  # approx_kl > KL_TARGET * 4 면 해당 업데이트 early stop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = "best_model.pt"
FINAL_MODEL_PATH = "final_model.pt"


# ─────────────────────────────────────────────────────────────
# ResNet Actor-Critic (+ aux delta head)
#  - obs: 1184 = board(64*16) + my_hand(5*16) + opp_hand(5*16)
#  - act_dim: 320 = 5 slots * 64 cells
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out)


class ActorCritic(nn.Module):
    """
    - board encoder: Conv2d(16→128) + ResBlock*5
    - hand encoder : Linear(160→128)
    - policy head  : board(32ch) + hand → logits(act_dim)
    - value head   : board(8ch)  + hand → scalar
    - delta head   : policy feature → per-action delta ([-1,1])
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.board_feat_dim = 64 * 16
        self.hand_feat_dim  = obs_dim - self.board_feat_dim  # 160

        # board encoder
        self.conv_in   = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
        self.bn_in     = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])

        # hand encoder
        self.fc_hand = nn.Sequential(
            nn.Linear(self.hand_feat_dim, 128),
            nn.ReLU(),
        )

        # policy head
        self.actor_conv = nn.Conv2d(128, 32, 1)
        self.actor_bn   = nn.BatchNorm2d(32)
        self.actor_in_dim = 32 * 8 * 8 + 128
        self.actor_fc   = nn.Linear(self.actor_in_dim, act_dim)

        # aux delta head
        self.delta_head = nn.Sequential(
            nn.Linear(self.actor_in_dim, self.actor_in_dim),
            nn.ReLU(),
            nn.Linear(self.actor_in_dim, act_dim),
            nn.Tanh(),   # label을 [-1,1]로 squash해서 학습
        )

        # value head
        self.critic_conv = nn.Conv2d(128, 8, 1)
        self.critic_bn   = nn.BatchNorm2d(8)
        self.critic_in_dim = 8 * 8 * 8 + 128
        self.critic_fc1  = nn.Linear(self.critic_in_dim, 256)
        self.critic_fc2  = nn.Linear(256, 1)

    def encode_board_hand(self, obs: torch.Tensor):
        """
        obs: [B, obs_dim]
          - 앞 1024: board  (64*16)
          - 뒤 160 : hands  (5*16 + 5*16)
        """
        B = obs.size(0)
        board_flat = obs[:, :self.board_feat_dim]      # [B, 1024]
        hand_flat  = obs[:, self.board_feat_dim:]      # [B, 160]

        board = board_flat.view(B, 64, 16).permute(0, 2, 1).contiguous()
        board = board.view(B, 16, 8, 8)

        x = self.conv_in(board)
        x = self.bn_in(x)
        x = F.relu(x)
        x = self.res_blocks(x)

        h = self.fc_hand(hand_flat)
        return x, h

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, obs_dim]
        return:
          logits     : [B, act_dim]
          value      : [B]
          delta_pred : [B, act_dim]
        """
        x, h = self.encode_board_hand(obs)
        B = obs.size(0)

        # policy path
        pol = self.actor_conv(x)
        pol = self.actor_bn(pol)
        pol = F.relu(pol)
        pol = pol.view(B, -1)
        pol = torch.cat([pol, h], dim=1)      # [B, actor_in_dim]

        logits = self.actor_fc(pol)
        delta_pred = self.delta_head(pol)

        # value path
        val = self.critic_conv(x)
        val = self.critic_bn(val)
        val = F.relu(val)
        val = val.view(B, -1)
        val = torch.cat([val, h], dim=1)
        val = F.relu(self.critic_fc1(val))
        value = self.critic_fc2(val).squeeze(-1)

        return logits, value, delta_pred

    def get_action(self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor = None):
        """
        obs: [B, obs_dim]
        mask: [B, act_dim] (bool)
        """
        logits, value, _ = self.forward(obs)
        logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value

    def get_value(self, obs: torch.Tensor):
        _, value, _ = self.forward(obs)
        return value


# ─────────────────────────────────────────────────────────────
# 마스킹된 categorical 분포
# ─────────────────────────────────────────────────────────────

def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    VERY_NEG = -1e9
    if mask.dtype == torch.bool:
        mask_f = mask.float()
    else:
        mask_f = mask
    masked_logits = logits + (1.0 - mask_f) * VERY_NEG

    # 혹시 전체가 0인 row가 있으면, 원래 logits 그대로 사용
    invalid_rows = (mask_f.sum(dim=-1) == 0)
    if invalid_rows.any():
        masked_logits[invalid_rows] = logits[invalid_rows]

    return Categorical(logits=masked_logits)


# ─────────────────────────────────────────────────────────────
# PPO 업데이트 (GAE + KL early stop + aux delta)
# ─────────────────────────────────────────────────────────────

def update_ppo(model: ActorCritic, optimizer: torch.optim.Optimizer, memory: dict):
    if len(memory["obs"]) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    obs   = torch.tensor(np.array(memory["obs"]),   dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.array(memory["act"]),   dtype=torch.long,    device=DEVICE)
    logp_old = torch.tensor(np.array(memory["logp"]), dtype=torch.float32, device=DEVICE)
    rew   = torch.tensor(np.array(memory["rew"]),   dtype=torch.float32, device=DEVICE)
    mask  = torch.tensor(np.array(memory["mask"]),  dtype=torch.bool,    device=DEVICE)
    vals  = torch.tensor(np.array(memory["val"]),   dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(np.array(memory["done"]),  dtype=torch.bool,    device=DEVICE)
    delta_raw = torch.tensor(np.array(memory["delta"]), dtype=torch.float32, device=DEVICE)

    # GAE
    vals_next = torch.cat([vals[1:], torch.tensor([0.0], device=DEVICE)])
    deltas = rew + GAMMA * vals_next * (~dones) - vals

    gae = 0.0
    adv_list = []
    for delta, done in zip(reversed(deltas.cpu().numpy()), reversed(dones.cpu().numpy())):
        if done:
            gae = 0.0
        gae = delta + GAMMA * LAMBDA * gae
        adv_list.insert(0, gae)

    advantages = torch.tensor(adv_list, dtype=torch.float32, device=DEVICE)
    returns = advantages + vals
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # delta 라벨 정규화
    delta_labels = torch.tanh(delta_raw / DELTA_SCALE)

    batch_size = obs.size(0)
    indices = np.arange(batch_size)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    total_delta_loss  = 0.0
    total_approx_kl   = 0.0
    minibatch_count   = 0

    kl_stop_threshold = KL_TARGET * KL_STOP_FACTOR
    stop_early = False

    for _ in range(K_EPOCH):
        np.random.shuffle(indices)
        for start in range(0, batch_size, MINI_BATCH_SIZE):
            if stop_early:
                break
            idx = indices[start:start + MINI_BATCH_SIZE]
            if len(idx) == 0:
                continue

            obs_mb   = obs[idx]
            act_mb   = act[idx]
            logp_old_mb = logp_old[idx]
            adv_mb   = advantages[idx]
            ret_mb   = returns[idx]
            mask_mb  = mask[idx]
            delta_mb = delta_labels[idx]

            logits, val_pred, delta_all = model(obs_mb)
            dist = masked_categorical(logits, mask_mb)

            logp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((val_pred - ret_mb) ** 2).mean()

            delta_pred = delta_all.gather(1, act_mb.unsqueeze(-1)).squeeze(-1)
            delta_loss = ((delta_pred - delta_mb) ** 2).mean()

            loss = (
                actor_loss
                + VF_COEF * value_loss
                - ENT_COEF * entropy
                + DELTA_LOSS_COEF * delta_loss
            )

            approx_kl = (logp_old_mb - logp).mean().detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += actor_loss.detach().cpu().item()
            total_value_loss  += value_loss.detach().cpu().item()
            total_entropy     += entropy.detach().cpu().item()
            total_delta_loss  += delta_loss.detach().cpu().item()
            total_approx_kl   += approx_kl
            minibatch_count   += 1

            if approx_kl > kl_stop_threshold:
                stop_early = True
                break
        if stop_early:
            break

    if minibatch_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_policy_loss / minibatch_count,
        total_value_loss  / minibatch_count,
        total_entropy     / minibatch_count,
        total_delta_loss  / minibatch_count,
        total_approx_kl   / minibatch_count,
    )


# ─────────────────────────────────────────────────────────────
# 메인 학습 루프 (self-play + heuristic opponent)
# ─────────────────────────────────────────────────────────────

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[INFO] Training start on {DEVICE} (self-play + heuristic opponent)")

    env = ConnexionEnv(seed=SEED)
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}, action_size: {env.action_size}")

    obs_dim = env.obs_dim
    act_dim = env.action_size
    print(f"Obs dim = {obs_dim}, Action dim = {act_dim}")

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    opponent_model = None

    # rollout memory
    memory = {
        "obs": [],
        "act": [],
        "logp": [],
        "rew": [],
        "mask": [],
        "done": [],
        "val": [],
        "delta": [],
    }

    timestep = 0
    score_q = deque(maxlen=100)
    diff_q  = deque(maxlen=100)

    global_best_diff = -9999.0
    current_opp_best_diff = 0.0

    for ep in range(1, MAX_EPISODES + 1):
        # ── 1. 이번 에피소드 상대 종류 선정 ──
        env.set_heuristic_mode(False)
        env.set_greedy_prob(0.0)
        opponent_kind = "random"
        current_opp = None

        if ep >= SELF_PLAY_START:
            # 1) 휴리스틱 상대
            if ep >= HEURISTIC_START_EP and random.random() < HEURISTIC_PROB:
                env.set_heuristic_mode(True)
                opponent_kind = "heuristic"
            else:
                # 2) self-play / greedy 상대 섞기
                roll = random.random()
                if roll < 0.33 and opponent_model is not None:
                    current_opp = opponent_model
                    opponent_kind = "selfplay_best"
                elif roll < 0.66:
                    env.set_greedy_prob(1.0)
                    opponent_kind = "greedy_1.0"
                else:
                    env.set_greedy_prob(0.5)
                    opponent_kind = "greedy_0.5"
        else:
            # 초기에는 랜덤만
            env.set_greedy_prob(0.0)
            opponent_kind = "random"

        # ── 2. best_model 기준으로 opponent_model 교체 ──
        if ep > SELF_PLAY_START and (ep - UPDATE_OFFSET) % UPDATE_OPPONENT_INT == 0:
            print(f"\n[INFO] 🔄 Update opponent with best model at episode {ep}")
            if os.path.exists(BEST_MODEL_PATH):
                opponent_model = ActorCritic(obs_dim, act_dim).to(DEVICE)
                opponent_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
                opponent_model.eval()
                print("   -> Loaded BEST_MODEL_PATH")
            else:
                opponent_model = copy.deepcopy(model).to(DEVICE)
                opponent_model.eval()
                print("   -> BEST_MODEL_PATH not found, using current model as opponent")
            current_opp_best_diff = 0.0

        # ── 3. 에피소드 플레이 ──
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        info = {}

        while not done:
            mask = env.get_smart_action_mask()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action_t, logp_t, _, val_t = model.get_action(obs_t, mask_t)
            a = int(action_t.item())

            next_obs, rew, done, _, info = env.step(a, opp_model=current_opp, device=DEVICE)

            # memory에 저장
            memory["obs"].append(obs)
            memory["act"].append(a)
            memory["logp"].append(float(logp_t.item()))
            memory["rew"].append(float(rew))
            memory["mask"].append(mask)
            memory["done"].append(bool(done))
            memory["val"].append(float(val_t.item()))
            memory["delta"].append(float(info.get("delta", 0.0)))

            obs = next_obs
            ep_rew += rew
            timestep += 1

            # 일정 step마다 PPO 업데이트
            if timestep % UPDATE_TIMESTEP == 0:
                (
                    pol_loss,
                    val_loss,
                    avg_entropy,
                    delta_loss,
                    avg_kl,
                ) = update_ppo(model, optimizer, memory)

                # 메모리 비우기
                for k in memory:
                    memory[k] = []

                print(
                    f"[Update @ step {timestep}] "
                    f"pol_loss={pol_loss:.4f} val_loss={val_loss:.4f} "
                    f"delta_loss={delta_loss:.4f} "
                    f"entropy={avg_entropy:.4f} approx_kl={avg_kl:.4f}"
                )

        # 에피소드 종료 후 로그용 큐에 추가
        score_q.append(ep_rew)
        if "diff" in info:
            diff_q.append(info["diff"])

        # ── 4. 로그 및 best_model 저장 ──
        if ep % 50 == 0:
            avg_score = sum(score_q) / len(score_q) if score_q else 0.0
            avg_diff  = sum(diff_q)  / len(diff_q)  if diff_q  else 0.0

            print(
                f"[Ep {ep:05d}] "
                f"Score(100ep)={avg_score:.2f} "
                f"Diff(100ep)={avg_diff:.1f} "
                f"Opp={opponent_kind}"
            )

            save_trigger = False

            # Case A: 휴리스틱(강적) 상대일 때
            if opponent_kind == "heuristic":
                if avg_diff > global_best_diff:
                    global_best_diff = avg_diff
                    save_trigger = True
                    print(f"  -> 🛡️ New best vs heuristic: diff={avg_diff:.1f}")

            # Case B: self-play best 상대일 때 (이겨야 의미)
            elif opponent_kind == "selfplay_best":
                if avg_diff > current_opp_best_diff and avg_diff > 0:
                    current_opp_best_diff = avg_diff
                    save_trigger = True
                    print(f"  -> 👑 Beat best self: diff={avg_diff:.1f}")

            # 기타 상대에 대한 저장 조건은 필요시 추가

            if save_trigger:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  -> Saved new BEST model to {BEST_MODEL_PATH}")

    # 학습 종료 후 최종 모델 저장
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"[INFO] Training finished. Final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train()
