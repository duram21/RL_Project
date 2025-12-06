#!/usr/bin/env python3
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connexion_env_10 import ConnexionEnv  # 또는 connexion_env_hybrid.py 이름에 맞춰서 수정

torch.set_num_threads(1)
device = torch.device("cpu")

# ─────────────────────────────────────
# 1. ActorCritic (ppo_connexion_hybrid.py와 동일)
# ─────────────────────────────────────

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
    ppo_connexion_hybrid.py와 동일한 구조:
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

        # aux delta head (훈련 때만 사용, 여기서는 forward만 같이 로드)
        self.delta_head = nn.Sequential(
            nn.Linear(self.actor_in_dim, self.actor_in_dim),
            nn.ReLU(),
            nn.Linear(self.actor_in_dim, act_dim),
            nn.Tanh(),
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


# ─────────────────────────────────────
# 2. 보드/타일 유틸 (env와 동일한 순서)
# ─────────────────────────────────────

COLORS  = ['R', 'G', 'B', 'Y']
SYMBOLS = ['1', '2', '3', '4']
FORBIDDEN = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}

VALID_CELLS = []
for c in ['a', 'b', 'c', 'd', 'e', 'f']:
    for r in ['1', '2', '3', '4', '5', '6']:
        for s in ['-', '+']:
            if (c + r + s) not in FORBIDDEN:
                VALID_CELLS.append(c + r + s)

NUM_CELLS = len(VALID_CELLS)
CELL_TO_IDX = {s: i for i, s in enumerate(VALID_CELLS)}

def parse_tile(s: str):
    """문자열(R1, B3, X0 등)을 내부 타일 id(0~15 또는 None)로 변환"""
    if s == "X0":
        return None
    try:
        c_idx = COLORS.index(s[0])
        n_idx = SYMBOLS.index(s[1])
        return c_idx * 4 + n_idx
    except Exception:
        return None

def tile_to_str(tid: int) -> str:
    return COLORS[tid // 4] + SYMBOLS[tid % 4]


# ─────────────────────────────────────
# 3. 메인 (testing-tool 프로토콜)
# ─────────────────────────────────────

def main():
    my_hand  = []                # 내 손패 (tile id list)
    opp_hand = []                # 상대 손패 (대충 추적만)
    board    = [-1] * NUM_CELLS  # 각 칸 tile id, 없으면 -1

    env_sim = None               # ConnexionEnv 인스턴스 (관측/마스크/휴리스틱용)
    model   = None               # ActorCritic

    def select_action_pure_ppo():
        """
        현재 board / my_hand / opp_hand 를 env_sim에 싱크 →
        valid_action_mask + 휴리스틱 logits + PPO logits를 섞어서 액션 선택
        """
        nonlocal env_sim, model, my_hand, opp_hand, board

        # 1) env 상태 동기화
        env_sim.board      = board[:]
        env_sim.agent_hand = my_hand[:]
        env_sim.opp_hand   = opp_hand[:]
        env_sim.filled     = sum(1 for x in board if x != -1)
        env_sim.done       = False

        # 2) valid mask (순수하게 valid만)
        valid_mask = env_sim.get_valid_action_mask(is_opponent=False)  # np.bool_ array
        if not valid_mask.any() or len(my_hand) == 0:
            # 도저히 못 고르면: 첫 타일 + 첫 빈칸
            c_idx = 0
            for i in range(NUM_CELLS):
                if board[i] == -1:
                    c_idx = i
                    break
            return 0 * NUM_CELLS + c_idx

        # 3) 휴리스틱 logits (공격 + 수비 + 중앙 보너스)
        heur_logits = env_sim.get_heuristic_logits(is_opponent=False)   # np.ndarray [act_dim]
        heur_logits = heur_logits.copy()
        heur_logits[~valid_mask] = -1e9    # invalid 액션은 매우 작은 값

        # 4) PPO logits
        obs  = env_sim._get_obs(is_opponent=False)
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            ppo_logits, _, _ = model(obs_t)      # [1, act_dim]
            ppo_logits = ppo_logits.squeeze(0)   # [act_dim]

        # 5) 하이브리드 결합 (PPO + λ * 휴리스틱)
        LAMBDA_HEU = 0.3  # 0.2~0.5 정도 튜닝 가능
        heur_t   = torch.tensor(heur_logits, dtype=torch.float32, device=device)
        combined = ppo_logits + LAMBDA_HEU * heur_t

        # 6) valid mask로 마지막 필터링
        VERY_NEG = -1e9
        valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        combined = combined.masked_fill(~valid_mask_t, VERY_NEG)

        action = int(torch.argmax(combined).item())
        return action

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue

        cmd = parts[0]

        # READY: env + 모델 lazy 로딩
        if cmd == "READY":
            print("OK")
            sys.stdout.flush()

            if model is None:
                env_sim = ConnexionEnv()
                obs_dim = env_sim.obs_dim
                act_dim = env_sim.action_size

                model_local = ActorCritic(obs_dim, act_dim).to(device)

                # 여러 후보 경로 시도 (새로 학습한 duram2 우선)
                candidate_paths = [
                    # "best_model_duram2.pt",
                    
                    "best_model_duram.pt",
           
                ]

                loaded = False
                for path in candidate_paths:
                    if not os.path.exists(path):
                        continue
                    try:
                        state = torch.load(path, map_location=device)
                        missing, unexpected = model_local.load_state_dict(state, strict=False)
                        try:
                            sys.stderr.write(
                                f"# Debug FIRST: Loaded {path} missing={list(missing.keys())} "
                                f"unexpected={list(unexpected.keys())}\n"
                            )
                            sys.stderr.flush()
                        except Exception:
                            pass
                        loaded = True
                        break
                    except Exception as e:
                        try:
                            sys.stderr.write(
                                f"# Debug FIRST: [WARN] Failed to load {path}: {e}\n"
                            )
                            sys.stderr.flush()
                        except Exception:
                            pass

                if not loaded:
                    try:
                        sys.stderr.write(
                            "# Debug FIRST: [WARN] No valid model file found. Using random weights.\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

                model_local.eval()
                model = model_local

        # INIT: 초기 패 / 상대 패 / 보드 리셋
        elif cmd == "INIT":
            my_hand.clear()
            opp_hand.clear()
            board[:] = [-1] * NUM_CELLS

            tiles = parts[1:]
            my_tiles  = tiles[:5]
            opp_tiles = tiles[5:10]

            for s in my_tiles:
                tid = parse_tile(s)
                if tid is not None:
                    my_hand.append(tid)

            for s in opp_tiles:
                tid = parse_tile(s)
                if tid is not None:
                    opp_hand.append(tid)

            if env_sim is not None:
                env_sim.reset()
                env_sim.board      = board[:]
                env_sim.agent_hand = my_hand[:]
                env_sim.opp_hand   = opp_hand[:]
                env_sim.filled     = 0
                env_sim.done       = False
                env_sim.prev_diff  = 0.0

        # TIME: 우리 차례 → PPO(+휴리스틱)로 수 선택
        elif cmd == "TIME":
            if model is None or env_sim is None or len(my_hand) == 0:
                # 준비 안 되었거나 손패 없으면 fallback
                slot = 0
                c_idx = 0
                for i in range(NUM_CELLS):
                    if board[i] == -1:
                        c_idx = i
                        break
            else:
                action = select_action_pure_ppo()
                slot = action // NUM_CELLS
                c_idx = action % NUM_CELLS

                # 안전장치: invalid 나오면 fallback
                if c_idx < 0 or c_idx >= NUM_CELLS or board[c_idx] != -1:
                    c_idx = 0
                    for i in range(NUM_CELLS):
                        if board[i] == -1:
                            c_idx = i
                            break
                if slot < 0 or slot >= len(my_hand):
                    slot = 0

            tid = my_hand.pop(slot)
            board[c_idx] = tid

            cell_str = VALID_CELLS[c_idx]
            tile_str = tile_to_str(tid)
            print(f"PUT {cell_str} {tile_str}")
            sys.stdout.flush()

        # GET: 새 타일 뽑음
        elif cmd == "GET":
            if len(parts) >= 2:
                tid = parse_tile(parts[1])
                if tid is not None:
                    my_hand.append(tid)

        # OPP: 상대 수 정보
        # 예: OPP a3+ G4 Y4 0
        elif cmd == "OPP":
            if len(parts) >= 3:
                cell_str = parts[1]
                t_play   = parse_tile(parts[2])
                t_draw   = parse_tile(parts[3]) if len(parts) >= 4 else None

                c_idx = CELL_TO_IDX.get(cell_str, None)
                if c_idx is not None and t_play is not None:
                    board[c_idx] = t_play
                    if t_play in opp_hand:
                        opp_hand.remove(t_play)
                    elif opp_hand:
                        opp_hand.pop(0)
                if t_draw is not None:
                    opp_hand.append(t_draw)

        # FINISH: 게임 종료
        elif cmd == "FINISH":
            break

        # 그 외 명령은 무시
        else:
            continue


if __name__ == "__main__":
    main()
