#!/usr/bin/env python3
import sys
import os

# lazy-load globals
torch = None
np = None
ConnexionEnv = None
device = None


def main():
    global torch, np, ConnexionEnv, device

    # -------------------------------------------------
    # 0. 기본 보드/타일 유틸 (env와 동일한 셀 순서)
    # -------------------------------------------------
    colors  = ['R', 'G', 'B', 'Y']
    symbols = ['1', '2', '3', '4']

    FORBIDDEN = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}

    valid_cells = []
    for c in ['a', 'b', 'c', 'd', 'e', 'f']:
        for r in ['1', '2', '3', '4', '5', '6']:
            for s in ['-', '+']:
                if (c + r + s) not in FORBIDDEN:
                    valid_cells.append(c + r + s)

    num_cells = len(valid_cells)  # 64
    cell_to_idx = {s: i for i, s in enumerate(valid_cells)}

    def parse_tile(s: str):
        """문자열(R1, B3, X0 등)을 내부 타일 id(0~15 또는 None)로 변환"""
        if s == "X0":
            return None
        try:
            c_idx = colors.index(s[0])
            n_idx = symbols.index(s[1])
            return c_idx * 4 + n_idx
        except Exception:
            return None

    def tile_to_str(tid: int) -> str:
        return colors[tid // 4] + symbols[tid % 4]

    # -------------------------------------------------
    # 1. 상태 (testing-tool <-> 우리 코드 싱크용)
    # -------------------------------------------------
    my_hand  = []               # 내 손패 (tile id)
    opp_hand = []               # 상대 손패(대충 추적)
    board    = [-1] * num_cells # 각 칸에 놓인 tile id, 없으면 -1

    env_sim = None
    model   = None

    # -------------------------------------------------
    # 2. ActorCritic (학습 코드의 ResNet 구조와 1:1로 맞춤)
    # -------------------------------------------------
    def build_model_from_env(env):
        import torch.nn as nn
        import torch.nn.functional as F

        obs_dim = env.obs_dim
        act_dim = env.action_size

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

        class ActorCriticImpl(nn.Module):
            """
            학습 스크립트에 있던 ResNet ActorCritic과 동일한 구조:

              - conv_in, bn_in, res_blocks, fc_hand
              - actor_conv, actor_bn, actor_fc
              - delta_fc (aux head)
              - critic_conv, critic_bn, critic_fc1, critic_fc2
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
                self.actor_in_dim = 32 * 8 * 8 + 128  # 32*64 + 128 = 2176
                self.actor_fc   = nn.Linear(self.actor_in_dim, act_dim)

                # aux delta head (훈련에서만 사용, ckpt 키 이름: "delta_fc.*")
                self.delta_fc = nn.Sequential(
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
                B = obs.size(0)
                board_flat = obs[:, :self.board_feat_dim]        # [B, 1024]
                hand_flat  = obs[:, self.board_feat_dim:]        # [B, 160]

                board = board_flat.view(B, 64, 16).permute(0, 2, 1).contiguous()
                board = board.view(B, 16, 8, 8)

                x = self.conv_in(board)
                x = self.bn_in(x)
                x = F.relu(x)
                x = self.res_blocks(x)

                h = self.fc_hand(hand_flat)
                return x, h

            # env._opponent_model_move에서 쓰는 용도 (혹시 self-play에 쓸 경우 대비)
            def forward_shared(self, obs: torch.Tensor):
                x, h = self.encode_board_hand(obs)
                return x, h

            def forward(self, obs: torch.Tensor):
                x, h = self.encode_board_hand(obs)
                B = obs.size(0)

                # policy path
                pol = self.actor_conv(x)
                pol = self.actor_bn(pol)
                pol = F.relu(pol)
                pol = pol.view(B, -1)
                pol = torch.cat([pol, h], dim=1)

                logits = self.actor_fc(pol)
                delta_pred = self.delta_fc(pol)

                # value path
                val = self.critic_conv(x)
                val = self.critic_bn(val)
                val = F.relu(val)
                val = val.view(B, -1)
                val = torch.cat([val, h], dim=1)
                val = F.relu(self.critic_fc1(val))
                value = self.critic_fc2(val).squeeze(-1)

                return logits, value, delta_pred

        return ActorCriticImpl(obs_dim, act_dim)

    # -------------------------------------------------
    # 3. 순수 PPO 정책으로 행동 선택 (env에서 obs/mask 가져오기)
    # -------------------------------------------------
    def select_action_pure_ppo():
        """현재 board / my_hand / opp_hand 상태를 env_sim에 반영하고, PPO policy로 greedy action 선택"""
        nonlocal env_sim, model

        # env 내부 상태 동기화
        env_sim.board      = board[:]
        env_sim.agent_hand = my_hand[:]
        env_sim.opp_hand   = opp_hand[:]
        env_sim.filled     = sum(1 for x in board if x != -1)

        # 관측 + valid action mask
        obs  = env_sim._get_obs(is_opponent=False)
        mask = env_sim.get_valid_action_mask(is_opponent=False)

        if not mask.any():
            # 이론상 거의 없음. fallback: 아무 legal move나
            for i in range(num_cells):
                if board[i] == -1 and len(my_hand) > 0:
                    return 0 * num_cells + i
            return 0

        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = model(obs_t)
            VERY_NEG = -1e9
            masked_logits = logits.masked_fill(~mask_t, VERY_NEG)
            action = torch.argmax(masked_logits, dim=-1).item()

        return int(action)

    # -------------------------------------------------
    # 4. 프로토콜 루프
    # -------------------------------------------------
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue

        cmd = parts[0]

        # READY: env + 모델 초기화
        if cmd == "READY":
            print("OK")
            sys.stdout.flush()

            if model is None:
                import torch as t
                import numpy as n
                from connexion_env_10 import ConnexionEnv as CE  # ⬅️ 네 env 파일 이름에 맞게 수정

                global torch, np, ConnexionEnv, device
                torch = t
                np = n
                ConnexionEnv = CE

                torch.set_num_threads(1)
                device = torch.device("cpu")

                # env 하나 만들어서 차원 정보 사용
                env_local = ConnexionEnv()
                # 혹시 휴리스틱 모드 켜져 있으면 끄기
                if hasattr(env_local, "set_heuristic_mode"):
                    try:
                        env_local.set_heuristic_mode(False)
                    except Exception:
                        pass
                env_sim = env_local

                # ResNet ActorCritic 생성
                model_local = build_model_from_env(env_sim).to(device)

                # ckpt 후보들 (네가 실제로 저장한 파일명에 맞게 수정)
                candidate_paths = [

                    "best_model_duram.pt",
                ]

                loaded = False
                for path in candidate_paths:
                    if not os.path.exists(path):
                        continue
                    try:
                        state = torch.load(path, map_location=device)
                        # 구조가 정확히 같으면 strict=True로 문제 없이 로딩됨
                        model_local.load_state_dict(state)
                        loaded = True
                        try:
                            sys.stderr.write(f"# Debug FIRST: Loaded model from {path}\n")
                            sys.stderr.flush()
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        try:
                            sys.stderr.write(f"# Debug FIRST: [WARN] Failed to load {path}: {e}\n")
                            sys.stderr.flush()
                        except Exception:
                            pass

                if not loaded:
                    try:
                        sys.stderr.write("# Debug FIRST: [WARN] No valid model file found. Using random weights.\n")
                        sys.stderr.flush()
                    except Exception:
                        pass

                model_local.eval()
                model = model_local

        # INIT: 초기 패 / 상대 패 / 보드 리셋
        elif cmd == "INIT":
            my_hand.clear()
            opp_hand.clear()
            board[:] = [-1] * num_cells

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
                env_sim.agent_hand = my_hand[:]
                env_sim.opp_hand   = opp_hand[:]
                env_sim.board      = board[:]
                env_sim.filled     = 0

        # TIME: 우리 차례 → 순수 PPO 정책으로 수 선택
        elif cmd == "TIME":
            if (model is None) or (env_sim is None) or (len(my_hand) == 0):
                # 준비 전이거나 손패가 없으면: 아무 legal move
                c_idx = 0
                for i in range(num_cells):
                    if board[i] == -1:
                        c_idx = i
                        break
                slot = 0
            else:
                action = select_action_pure_ppo()
                slot = action // num_cells
                c_idx = action % num_cells

                # 안전장치: 잘못된 액션이면 fallback
                if c_idx < 0 or c_idx >= num_cells or board[c_idx] != -1:
                    c_idx = 0
                    for i in range(num_cells):
                        if board[i] == -1:
                            c_idx = i
                            break
                if slot < 0 or slot >= len(my_hand):
                    slot = 0

            tid = my_hand.pop(slot)
            board[c_idx] = tid

            cell_str = valid_cells[c_idx]
            tile_str = tile_to_str(tid)
            print(f"PUT {cell_str} {tile_str}")
            sys.stdout.flush()

        # GET: 내가 새 타일을 뽑음
        elif cmd == "GET":
            if len(parts) >= 2:
                tid = parse_tile(parts[1])
                if tid is not None:
                    my_hand.append(tid)

        # OPP: 상대 수 정보
        elif cmd == "OPP":
            if len(parts) >= 3:
                cell_str = parts[1]
                t_play   = parse_tile(parts[2])
                t_draw   = parse_tile(parts[3]) if len(parts) >= 4 else None

                c_idx = cell_to_idx.get(cell_str, None)
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

        # 기타 명령은 무시
        else:
            continue


if __name__ == "__main__":
    main()
