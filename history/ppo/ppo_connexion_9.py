import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from connexion_env_9 import ConnexionEnv  # 휴리스틱 teacher + opponent 내장 버전

# ------------------------------
# PPO 하이퍼파라미터
# ------------------------------
TOTAL_UPDATES    = 1000          # 전체 업데이트 횟수
STEPS_PER_UPDATE = 2048         # rollout 길이 (한 번에 모을 step 수)
GAMMA            = 0.99
LAMBDA           = 0.95
CLIP_EPS         = 0.15         # PPO clip epsilon
LR               = 2.5e-4
EPOCHS           = 4
MINIBATCH_SIZE   = 256
ENT_COEF         = 0.005        # entropy bonus 계수
VF_COEF          = 0.5          # value loss 계수
MAX_GRAD_NORM    = 0.5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Aux delta head 관련 하이퍼파라미터 ---
DELTA_SCALE      = 20.0         # my_delta 정규화 스케일
DELTA_LOSS_COEF  = 0.1          # aux delta loss 가중치

MODEL_PATH          = "ppo_connexion_hybrid_resnet_final.pt"
BEST_MODEL_PATH     = "ppo_connexion_hybrid_resnet_best.pt"
CHECKPOINT_INTERVAL = 50        # N업데이트마다 체크포인트
EVAL_INTERVAL       = 50        # 몇 업데이트마다 평가할지
EVAL_EPISODES       = 100       # 평가 때 몇 판 돌려볼지


# -------------------------------------------------
# residual_coef 스케줄 (휴리스틱 베이스 + PPO 살짝)
# -------------------------------------------------
def get_residual_coef(update: int) -> float:
    """
    휴리스틱을 강하게 유지하고, PPO는 보정 정도로만 쓰는 스케줄.

    - 1 ~ 80 업데이트: 0.10 -> 0.30 (조금씩 증가)
    - 81 이후: 0.30 고정

    즉, 항상 최소 70% 이상은 휴리스틱 쪽이 영향을 주도록 설계.
    (residual_coef == 0 이 되면 gradient가 거의 안 흐르니 0은 피함)
    """
    if update <= 80:
        # 선형으로 0.10 -> 0.30
        return 0.10 + 0.20 * (update - 1) / max(1, 80 - 1)
    else:
        return 0.30


# ------------------------------
# ResNet 블록
# ------------------------------
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
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return torch.relu(x + out)


# ------------------------------
# Actor-Critic 네트워크 (+ delta head)
#  - 보드: 64셀 × 16차원 one-hot → (B,16,8,8) 로 reshape
#  - 손패: (내 5장 + 상대 5장) → 총 5*16*2 = 160차원
# ------------------------------
class ActorCritic(nn.Module):
    """
    구조 (기존 ResNet 버전 기반 + delta head 추가):

      공유 trunk:
        - board: Conv2d(16→128) + BN + ReLU + ResBlock*5
        - hand : Linear(160→128) + ReLU

      policy head:
        - board feature: Conv2d(128→32) + BN + ReLU → flatten
        - concat(hand_feat) → Linear → logits(act_dim)

      value head:
        - board feature: Conv2d(128→8) + BN + ReLU → flatten
        - concat(hand_feat) → Linear → Linear(1)

      delta head:
        - policy feature와 같은 입력(pol_feat)에 Linear → Tanh(act_dim)
          (각 액션에 대한 my_delta 예측, [-1,1] 범위)
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Connexion 기준: num_cells=64, board_feat_dim=64*16=1024
        self.board_feat_dim = 64 * 16
        self.hand_feat_dim  = obs_dim - self.board_feat_dim   # 5*16 + 5*16 = 160 일 것

        # board encoder
        self.conv_in   = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
        self.bn_in     = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])

        # hand encoder (내 손 + 상대 손 합친 160차원)
        self.fc_hand = nn.Sequential(
            nn.Linear(self.hand_feat_dim, 128),
            nn.ReLU(),
        )

        # policy head
        self.actor_conv = nn.Conv2d(128, 32, 1)
        self.actor_bn   = nn.BatchNorm2d(32)
        self.actor_in_dim = 32 * 8 * 8 + 128  # board feature + hand feature
        self.actor_fc   = nn.Linear(self.actor_in_dim, act_dim)

        # aux delta head (policy feature와 동일한 입력 사용)
        self.delta_fc   = nn.Sequential(
            nn.Linear(self.actor_in_dim, self.actor_in_dim),
            nn.ReLU(),
            nn.Linear(self.actor_in_dim, act_dim),
            nn.Tanh(),   # 라벨도 [-1,1]로 정규화해서 학습
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
          - 앞 1024: board (64 * 16)
          - 나머지: hand (5*16 + 5*16 = 160)
        """
        B = obs.size(0)
        board_flat = obs[:, :self.board_feat_dim]          # [B, 1024]
        hand_flat  = obs[:, self.board_feat_dim:]          # [B, 160]

        # board_flat: [B, 64*16] → [B, 64, 16] → [B, 16, 64] → [B,16,8,8]
        board = board_flat.view(B, 64, 16).permute(0, 2, 1).contiguous()
        board = board.view(B, 16, 8, 8)

        x = self.conv_in(board)
        x = self.bn_in(x)
        x = torch.relu(x)
        x = self.res_blocks(x)

        h = self.fc_hand(hand_flat)
        return x, h

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, obs_dim]
        return:
          logits     : [B, act_dim]
          value      : [B]
          delta_pred : [B, act_dim]  (각 액션에 대한 delta 예측)
        """
        x, h = self.encode_board_hand(obs)
        B = obs.size(0)

        # policy path
        pol = self.actor_conv(x)
        pol = self.actor_bn(pol)
        pol = torch.relu(pol)
        pol = pol.view(B, -1)           # [B, 32*8*8]
        pol = torch.cat([pol, h], dim=1)  # [B, actor_in_dim]

        logits = self.actor_fc(pol)
        delta_pred = self.delta_fc(pol)

        # value path
        val = self.critic_conv(x)
        val = self.critic_bn(val)
        val = torch.relu(val)
        val = val.view(B, -1)           # [B, 8*8*8]
        val = torch.cat([val, h], dim=1)
        val = torch.relu(self.critic_fc1(val))
        value = self.critic_fc2(val).squeeze(-1)   # [B]

        return logits, value, delta_pred


# ------------------------------
# 마스킹된 categorical 분포
# ------------------------------
def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """
    logits: [B, A]
    mask  : [B, A] (0/1 또는 bool)

    mask == 0 인 곳은 선택 불가하게 아주 큰 음수로 막는다.
    """
    VERY_NEG = -1e9

    if mask.dtype == torch.bool:
        mask_f = mask.float()
    else:
        mask_f = mask

    masked_logits = logits + (1.0 - mask_f) * VERY_NEG

    # 혹시 어떤 row는 mask가 전부 0일 수도 있으니(이론상 거의 없음)
    invalid_rows = (mask_f.sum(dim=-1) == 0)
    if invalid_rows.any():
        masked_logits[invalid_rows] = logits[invalid_rows]

    return Categorical(logits=masked_logits)


# ------------------------------
# Rollout 수집 (휴리스틱 + PPO residual)
# ------------------------------
def collect_rollout(env: ConnexionEnv, model: ActorCritic, residual_coef: float):
    """
    env에서 STEPS_PER_UPDATE 만큼 데이터를 모아서
    PPO 학습에 쓸 버퍼를 반환.

    residual_coef:
      combined_logits = (1-residual_coef)*heur_logits + residual_coef*policy_logits
      형태로 섞어서 행동을 샘플링.
    """
    obs_dim = env._get_obs().shape[0]
    act_dim = env.action_size

    obs_buf   = np.zeros((STEPS_PER_UPDATE, obs_dim), dtype=np.float32)
    act_buf   = np.zeros((STEPS_PER_UPDATE,), dtype=np.int64)
    logp_buf  = np.zeros((STEPS_PER_UPDATE,), dtype=np.float32)
    rew_buf   = np.zeros((STEPS_PER_UPDATE,), dtype=np.float32)
    val_buf   = np.zeros((STEPS_PER_UPDATE,), dtype=np.float32)
    done_buf  = np.zeros((STEPS_PER_UPDATE,), dtype=np.float32)
    mask_buf  = np.zeros((STEPS_PER_UPDATE, act_dim), dtype=np.float32)
    delta_buf = np.zeros((STEPS_PER_UPDATE,), dtype=np.float32)  # aux delta 라벨

    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    episode_returns_for_logging = []

    model.eval()

    for t in range(STEPS_PER_UPDATE):
        obs_buf[t] = obs

        # 현재 상태에서 valid action mask, heuristic logits 가져오기
        action_mask = env.get_valid_action_mask()       # np.bool array
        heur_logits = env.get_heuristic_logits()        # np.float32, shape (act_dim,)

        mask_buf[t] = action_mask.astype(np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        heur_logits_t = torch.tensor(heur_logits, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value, _ = model(obs_t)

            # 휴리스틱 + PPO logits 섞기
            combined_logits = (1.0 - residual_coef) * heur_logits_t + residual_coef * policy_logits

            dist = masked_categorical(combined_logits, mask_t)
            action = dist.sample()
            logp = dist.log_prob(action)

        action_int = int(action.item())

        act_buf[t]  = action_int
        logp_buf[t] = logp.item()
        val_buf[t]  = value.squeeze(0).item()

        # env.step 전에 현재 상태에서의 my_delta 추정값을 라벨로 저장
        my_delta_est = env.estimate_my_delta_for_action(action_int)  # scalar
        delta_buf[t] = my_delta_est

        next_obs, reward, done, truncated, info = env.step(action_int)
        rew_buf[t]  = reward
        done_buf[t] = float(done)

        ep_return += reward
        ep_len += 1

        if done:
            episode_returns_for_logging.append(ep_return)
            ep_return = 0.0
            ep_len = 0
            next_obs, _ = env.reset()

        obs = next_obs

    # 마지막 상태 value (부트스트랩)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        _, last_val, _ = model(obs_t)
    last_val = last_val.item()

    model.train()

    return (
        obs_buf,
        act_buf,
        logp_buf,
        rew_buf,
        val_buf,
        done_buf,
        mask_buf,
        delta_buf,                      # aux delta 라벨
        last_val,
        episode_returns_for_logging,
    )


# ------------------------------
# GAE advantage & returns 계산
# ------------------------------
def compute_gae(rews, vals, dones, last_val, gamma=GAMMA, lam=LAMBDA):
    """
    rews, vals, dones: 길이 T (STEPS_PER_UPDATE)
    last_val: 마지막 상태의 V(s_T)
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    vals_ext = np.append(vals, last_val)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * vals_ext[t + 1] * nonterminal - vals_ext[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + vals
    return adv, returns


# ------------------------------
# 평가 루프 (랜덤 상대와 N판)
# ------------------------------
def evaluate(model: ActorCritic, n_episodes: int = EVAL_EPISODES, residual_coef_eval: float = 0.30):
    """
    평가 환경은 고정: my_role=FIRST, opponent=random
    → "랜덤 상대로 얼마나 이기는지"를 지표로 사용.

    여기서도 rollout 때와 동일하게:
      combined_logits = (1-res)*heur + res*pi
    방식으로 greedy 정책 사용.
    """
    env = ConnexionEnv(my_role=0, opponent_policy="random")
    model.eval()

    final_diffs = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            last_info = None

            while not done:
                mask = env.get_valid_action_mask()
                if not mask.any():
                    break

                heur_logits = env.get_heuristic_logits()  # np.float32, (A,)

                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                heur_logits_t = torch.tensor(heur_logits, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                policy_logits, _, _ = model(obs_t)

                # 휴리스틱 + PPO 섞은 후, masked argmax
                combined_logits = (1.0 - residual_coef_eval) * heur_logits_t + residual_coef_eval * policy_logits

                VERY_NEG = -1e9
                masked_logits = combined_logits + (1.0 - mask_t) * VERY_NEG
                action = torch.argmax(masked_logits, dim=-1).item()

                obs, reward, done, truncated, info = env.step(action)
                last_info = info

            if last_info is not None:
                # ConnexionEnv에서 반환하는 키 이름에 맞추기
                # first: 우리(에이전트) 점수, second: 상대 점수
                my_score   = last_info.get("first", 0.0)
                opp_score  = last_info.get("second", 0.0)
                final_diffs.append(my_score - opp_score)
            else:
                final_diffs.append(0.0)


    final_diffs = np.array(final_diffs, dtype=np.float32)
    mean_diff = float(final_diffs.mean())
    winrate = float((final_diffs > 0).mean())  # 최종 점수차 기준 승률

    return mean_diff, winrate


# ------------------------------
# 메인 학습 루프
# ------------------------------
def train():
    """
    커리큘럼 스케줄:
      - update  1 ~  80: opponent = random
      - update 81 ~ 140: opponent = mixed (random/greedy 50%)
      - update 141 ~ ...: opponent = greedy

    정책은 항상:
      combined_logits = (1-res)*heur + res*pi
    로 행동하며, res는 get_residual_coef(update)로 결정.
    """

    # 초기에는 random 상대로 env 생성 (base_opponent_policy만 중간에 바꿔줌)
    env = ConnexionEnv(my_role=0, opponent_policy="random")
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape, " action_size:", env.action_size)

    # obs/action 차원
    obs_sample, _ = env.reset()
    obs_dim = obs_sample.shape[0]
    act_dim = env.action_size

    print(f"Obs dim = {obs_dim}, Action dim = {act_dim}")

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    all_episode_returns = []
    best_mean_diff = -1e9   # 지금까지 평가 기준 최고 성능

    for update in range(1, TOTAL_UPDATES + 1):
        # --------- 커리큘럼: 상대 난이도 조절 ---------
        if update <= 80:
            # 완전 랜덤 상대
            env.base_opponent_policy = "random"
        elif update <= 140:
            # random / greedy 혼합
            env.base_opponent_policy = "mixed"
            env.opponent_mix_greedy_prob = 0.5
        else:
            # 완전 greedy 상대
            env.base_opponent_policy = "greedy"

        # --------- residual_coef (휴리스틱 vs PPO 비율) ---------
        residual_coef = get_residual_coef(update)

        (
            obs_buf,
            act_buf,
            logp_buf,
            rew_buf,
            val_buf,
            done_buf,
            mask_buf,
            delta_buf,         # aux delta 라벨
            last_val,
            ep_returns,
        ) = collect_rollout(env, model, residual_coef)

        global_step += STEPS_PER_UPDATE
        all_episode_returns.extend(ep_returns)

        # GAE + returns
        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_val)
        # advantage 정규화 (학습 안정성)
        adv_mean = adv_buf.mean()
        adv_std = adv_buf.std() + 1e-8
        adv_buf = (adv_buf - adv_mean) / adv_std

        # delta 라벨 정규화 ([-1,1]로 squash)
        delta_labels = np.tanh(delta_buf / DELTA_SCALE)

        # numpy -> torch
        obs_t      = torch.tensor(obs_buf,        dtype=torch.float32, device=DEVICE)
        act_t      = torch.tensor(act_buf,        dtype=torch.long,   device=DEVICE)
        old_logp_t = torch.tensor(logp_buf,       dtype=torch.float32, device=DEVICE)
        adv_t      = torch.tensor(adv_buf,        dtype=torch.float32, device=DEVICE)
        ret_t      = torch.tensor(ret_buf,        dtype=torch.float32, device=DEVICE)
        mask_t     = torch.tensor(mask_buf,       dtype=torch.float32, device=DEVICE)
        delta_t    = torch.tensor(delta_labels,   dtype=torch.float32, device=DEVICE)

        # 미니배치용 index 셔플
        batch_size = STEPS_PER_UPDATE
        idxs = np.arange(batch_size)

        # 로깅용 누적 변수
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_approx_kl   = 0.0
        total_delta_loss  = 0.0
        minibatch_count   = 0

        for epoch in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                end   = start + MINIBATCH_SIZE
                mb_idx = idxs[start:end]

                obs_mb      = obs_t[mb_idx]
                act_mb      = act_t[mb_idx]
                old_logp_mb = old_logp_t[mb_idx]
                adv_mb      = adv_t[mb_idx]
                ret_mb      = ret_t[mb_idx]
                mask_mb     = mask_t[mb_idx]
                delta_mb    = delta_t[mb_idx]

                # forward
                policy_logits, value, delta_all = model(obs_mb)

                # rollout 때와 동일하게 heuristic logits은 없음
                # (rollout에서 이미 combined_logits 기준으로 old_logp 저장)
                # 여기서는 policy_logits만으로 새 logp 계산하고,
                # old_logp는 "combined policy"에서 나왔던 값으로 취급.
                #
                # 즉, gradient는 policy_logits 쪽으로만 흐르고,
                # heuristic은 고정된 teacher 역할로 남음.
                dist   = masked_categorical(policy_logits, mask_mb)
                logp   = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(logp - old_logp_mb)

                # clipped surrogate objective
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = ((value - ret_mb) ** 2).mean()

                # aux delta loss: delta_head가 예측한 값 vs 라벨
                # delta_all: [B, act_dim], act_mb: [B]
                delta_pred = delta_all.gather(1, act_mb.unsqueeze(-1)).squeeze(-1)
                delta_loss = ((delta_pred - delta_mb) ** 2).mean()

                # 총 loss
                loss = (
                    policy_loss
                    + VF_COEF * value_loss
                    - ENT_COEF * entropy
                    + DELTA_LOSS_COEF * delta_loss
                )

                # approx KL (old || new) ≈ E[logp_old - logp_new]
                approx_kl = (old_logp_mb - logp).mean().detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_policy_loss += policy_loss.detach().cpu().item()
                total_value_loss  += value_loss.detach().cpu().item()
                total_entropy     += entropy.detach().cpu().item()
                total_delta_loss  += delta_loss.detach().cpu().item()
                total_approx_kl   += approx_kl
                minibatch_count   += 1

        # 모니터링: 최근 episode return 평균 (20 / 100 / 전체)
        if len(all_episode_returns) > 0:
            recent20  = all_episode_returns[-20:] if len(all_episode_returns) >= 20 else all_episode_returns
            recent100 = all_episode_returns[-100:] if len(all_episode_returns) >= 100 else all_episode_returns

            avg20   = float(np.mean(recent20))
            avg100  = float(np.mean(recent100))
            avg_all = float(np.mean(all_episode_returns))
        else:
            avg20 = avg100 = avg_all = 0.0

        avg_policy_loss = total_policy_loss / max(1, minibatch_count)
        avg_value_loss  = total_value_loss  / max(1, minibatch_count)
        avg_entropy     = total_entropy     / max(1, minibatch_count)
        avg_kl          = total_approx_kl   / max(1, minibatch_count)
        avg_delta_loss  = total_delta_loss  / max(1, minibatch_count)

        print(
            f"[Update {update}] step={global_step} eps={len(all_episode_returns)} "
            f"avg20={avg20:.2f} avg100={avg100:.2f} avg_all={avg_all:.2f} "
            f"pol_loss={avg_policy_loss:.4f} val_loss={avg_value_loss:.4f} "
            f"delta_loss={avg_delta_loss:.4f} "
            f"entropy={avg_entropy:.4f} approx_kl={avg_kl:.4f} "
            f"res_coef={residual_coef:.3f}"
        )

        # 주기적인 체크포인트 저장 (상태 백업용)
        if update % CHECKPOINT_INTERVAL == 0:
            ckpt_path = f"ppo_connexion_hybrid_resnet_ckpt_{update:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

        # 주기적으로 성능 평가 + 베스트 모델 저장 (항상 랜덤 상대 기준)
        if update % EVAL_INTERVAL == 0:
            # 평가 시에도 heuristic+PPO(0.3) 섞어서 사용
            mean_diff, winrate = evaluate(model, n_episodes=EVAL_EPISODES, residual_coef_eval=0.30)
            print(
                f"[Eval @ update {update}] "
                f"mean_score_diff={mean_diff:.2f}, winrate={winrate*100:.1f}%"
            )

            if mean_diff > best_mean_diff:
                best_mean_diff = mean_diff
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"[Eval] New best model saved to {BEST_MODEL_PATH} (mean_diff={mean_diff:.2f})")

    # 최종 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training finished. Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
