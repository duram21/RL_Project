# connexion_env_hybrid.py
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

# ─────────────────────────────────────────────────────────────
# 기본 정의 (보드 / 타일 / 인접 리스트)
# ─────────────────────────────────────────────────────────────

class Column(Enum):
    A = "a"; B = "b"; C = "c"; D = "d"; E = "e"; F = "f"

class Row(Enum):
    _1 = "1"; _2 = "2"; _3 = "3"; _4 = "4"; _5 = "5"; _6 = "6"

class Sign(Enum):
    MINUS = "-"; PLUS = "+"

class Color(Enum):
    R = "R"; G = "G"; B = "B"; Y = "Y"

class Symbol(Enum):
    _1 = "1"; _2 = "2"; _3 = "3"; _4 = "4"

# 금지된 칸
FORBIDDEN_STR = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}


@dataclass(frozen=True)
class Cell:
    col: Column
    row: Row
    sign: Sign

    def __str__(self) -> str:
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
        return str(self) not in FORBIDDEN_STR

    def isAdjacent(self, other: "Cell") -> bool:
        """Connexion 규칙에 맞는 인접 판정"""
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        # -, + 사이의 인접 규칙
        if self.sign == Sign.MINUS and other.sign == Sign.PLUS:
            return (dr == 0 and dc == 0) or (dr == 0 and dc == -1) or (dr == -1 and dc == 0)
        if self.sign == Sign.PLUS and other.sign == Sign.MINUS:
            return (dr == 0 and dc == 0) or (dr == 0 and dc == 1) or (dr == 1 and dc == 0)
        return False


@dataclass(frozen=True)
class Tile:
    color: Color
    symbol: Symbol

    def __str__(self) -> str:
        return f"{self.color.value}{self.symbol.value}"


# 전체 칸 / 인접 리스트 / 타일 ID 매핑
ALL_CELLS: List[Cell] = []
for col in Column:
    for row in Row:
        for sign in Sign:
            c = Cell(col, row, sign)
            if c.isValid():
                ALL_CELLS.append(c)

NUM_CELLS: int = len(ALL_CELLS)          # 보드 칸 수 (64여야 함)
CELL_INDEX: Dict[str, int] = {str(c): i for i, c in enumerate(ALL_CELLS)}

ADJ: List[List[int]] = [[] for _ in range(NUM_CELLS)]
for i, ci in enumerate(ALL_CELLS):
    for j, cj in enumerate(ALL_CELLS):
        if i != j and ci.isAdjacent(cj):
            ADJ[i].append(j)

COLOR_LIST = [Color.R, Color.G, Color.B, Color.Y]
SYMBOL_LIST = [Symbol._1, Symbol._2, Symbol._3, Symbol._4]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LIST)}
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOL_LIST)}

def tile_to_id(t: Tile) -> int:
    return COLOR_TO_IDX[t.color] * 4 + SYMBOL_TO_IDX[t.symbol]

def id_to_tile(tid: int) -> Tile:
    return Tile(COLOR_LIST[tid // 4], SYMBOL_LIST[tid % 4])


# ─────────────────────────────────────────────────────────────
# Connexion Env (self-play + 내부 휴리스틱/greedy 내장)
# 보드 1024 + 내 패 80 + 상대 패 80 = obs_dim=1184
# 액션: 5장 슬롯 × 64 칸 = 320
# ─────────────────────────────────────────────────────────────

class ConnexionEnv:
    """
    - 에이전트는 항상 "first" 기준으로 점수 계산.
    - step(action, opp_model=None, device=None):
        * agent 수 두고
        * opp_model 있으면 PPO opponent (self-play)
        * 없으면 내부 random/greedy/heuristic 사용
        * 이후 스코어 diff 변화(delta)를 reward로 사용
    """

    def __init__(self,
                 seed: int = 0,
                 my_role: int = 0,                 # 호환용 (사용 안 해도 무방)
                 opponent_policy: str = "random"   # "random", "greedy", "mixed", "heuristic"
                 ):
        self.rng = np.random.RandomState(seed)
        self.num_cells = NUM_CELLS
        self.action_size = self.num_cells * 5  # 5 hand slots
        # Board(64*16) + my hand(5*16) + opp hand(5*16)
        self.obs_dim = self.num_cells * 16 + 5 * 16 + 5 * 16

        # 보상 관련
        self.reward_scale = 40.0
        self.win_bonus = 1.0

        # 상대 정책 관련
        self.base_opponent_policy = opponent_policy
        self.opponent_mix_greedy_prob = 0.5    # "mixed"일 때 greedy 확률
        self.greedy_prob = 0.0                 # 예전 코드 호환용

        # 휴리스틱 플래그 (예전 API 호환)
        self.use_heuristic = False

        # 게임 상태
        self.board: List[int] = []
        self.agent_hand: List[int] = []
        self.opp_hand: List[int] = []
        self.agent_deck: List[int] = []
        self.opp_deck: List[int] = []
        self.filled: int = 0
        self.done: bool = False
        self.prev_diff: float = 0.0
        self.last_delta: float = 0.0

        self.reset()

    # ───────── 외부 설정 메서드 (예전 코드 호환 + 새 코드) ─────────

    def set_greedy_prob(self, prob: float):
        """예전 코드에서 쓰던 greedy_prob (0 ~ 1)."""
        self.greedy_prob = float(np.clip(prob, 0.0, 1.0))

    def set_heuristic_mode(self, use_heuristic: bool, is_first_for_heuristic: bool = False):
        """
        예전 코드와의 호환을 위해 남겨둔 함수.
        - use_heuristic=True이면, 상대는 강한 greedy 휴리스틱만 사용.
        """
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.base_opponent_policy = "heuristic"

    def set_base_opponent_policy(self, policy: str):
        """새 코드용: 'random' / 'greedy' / 'mixed' / 'heuristic'"""
        self.base_opponent_policy = policy

    # ───────── reset / 관측 / 마스크 ─────────

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        # 풀 덱 생성 (각 타일 4장, 총 64장)
        full_deck_tiles = [Tile(c, s) for c in COLOR_LIST for s in SYMBOL_LIST for _ in range(4)]
        self.rng.shuffle(full_deck_tiles)
        full_deck = [tile_to_id(t) for t in full_deck_tiles]

        # 단순 분배: 27 + 5 씩
        self.agent_deck = full_deck[:27]
        self.opp_deck   = full_deck[27:54]
        self.agent_hand = full_deck[54:59]
        self.opp_hand   = full_deck[59:64]

        self.board  = [-1] * self.num_cells
        self.filled = 0
        self.done   = False
        self.prev_diff = 0.0
        self.last_delta = 0.0

        return self._get_obs(), {}

    def _get_obs(self, is_opponent: bool = False) -> np.ndarray:
        """
        관측:
          - Board: one-hot (64 * 16 = 1024)
          - My hand: 5 * 16
          - Opp hand: 5 * 16
        is_opponent=True이면, 입장 반대로 (opp_hand ↔ agent_hand)
        """
        my_target_hand  = self.opp_hand if is_opponent else self.agent_hand
        opp_target_hand = self.agent_hand if is_opponent else self.opp_hand

        board_feat = np.zeros((self.num_cells, 16), dtype=np.float32)
        for idx, tid in enumerate(self.board):
            if tid != -1:
                board_feat[idx, tid] = 1.0

        my_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(my_target_hand):
            if slot < 5:
                my_hand_feat[slot, tid] = 1.0

        opp_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(opp_target_hand):
            if slot < 5:
                opp_hand_feat[slot, tid] = 1.0

        return np.concatenate([
            board_feat.flatten(),
            my_hand_feat.flatten(),
            opp_hand_feat.flatten(),
        ])

    def get_valid_action_mask(self, is_opponent: bool = False) -> np.ndarray:
        """
        액션: hand_slot(0~4) * num_cells + cell_idx
        손에 있는 타일 수만큼만 슬롯 사용.
        """
        target_hand = self.opp_hand if is_opponent else self.agent_hand
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done:
            return mask

        for slot in range(len(target_hand)):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    mask[slot * self.num_cells + ci] = True
        return mask

    def get_smart_action_mask(self) -> np.ndarray:
        valid_mask = self.get_valid_action_mask()
        if valid_mask.sum() == 0:
            return valid_mask

        heur = self.get_heuristic_logits(is_opponent=False)  # 공격+수비
        heur[~valid_mask] = -1e9

        max_h = np.max(heur)
        if max_h <= -1e8:
            return valid_mask

        threshold = max_h * 0.7
        mask = (heur >= threshold) & valid_mask

        if not mask.any():
            return valid_mask
        return mask


    # ───────── 휴리스틱 teacher용 logits (옵션) ─────────

    def get_heuristic_logits(self, is_opponent: bool = False) -> np.ndarray:
        """
        각 액션에 대한 휴리스틱 점수를 "logit"처럼 반환.
        - 공격(local_score) + 중앙 가중치 + 간단한 방어 점수 조합.
        - PPO residual teacher로 사용할 수 있음.
        """
        logits = np.full(self.action_size, -1e9, dtype=np.float32)
        if self.done:
            return logits

        if is_opponent:
            my_hand = self.opp_hand
            opp_hand = self.agent_hand
            is_first = False
        else:
            my_hand = self.agent_hand
            opp_hand = self.opp_hand
            is_first = True

        if len(my_hand) == 0:
            return logits

        for slot, tid in enumerate(my_hand):
            for c in range(self.num_cells):
                if self.board[c] != -1:
                    continue
                idx = slot * self.num_cells + c
                atk = self._local_score(c, tid, is_first=is_first)
                center_bonus = len(ADJ[c]) * 0.5

                # 간단한 방어 점수: 상대가 여기 뒀을 때 얻는 최대 local_score
                defense = 0
                for opp_tid in opp_hand:
                    defense = max(defense, self._local_score(c, opp_tid, is_first=not is_first))

                val = atk + center_bonus + 0.8 * defense
                logits[idx] = val

        return logits

    def estimate_my_delta_for_action(self, action_idx: int) -> float:
        """
        현재 state에서 주어진 action을 두었을 때
        (상대 수는 아직 두지 않았다고 가정) first-second diff가 얼마나 바뀌는지 추정.
        - 보드에만 가상으로 두고 diff 변화 계산 → label용으로 사용.
        """
        slot = action_idx // self.num_cells
        c_idx = action_idx % self.num_cells

        if slot >= len(self.agent_hand) or self.board[c_idx] != -1:
            return 0.0

        # 현재 diff
        _, _, curr_diff = self._compute_scores()

        # 가상 착수
        tid = self.agent_hand[slot]
        bak_board = list(self.board)
        bak_filled = self.filled

        self.board[c_idx] = tid
        self.filled += 1
        _, _, new_diff = self._compute_scores()

        # 복구
        self.board = bak_board
        self.filled = bak_filled

        return float(new_diff - curr_diff)

    # ───────── step / 내부 수 적용 ─────────

    def step(self, action: int, opp_model=None, device=None):
        """
        action: 우리의 액션 (slot * num_cells + cell_idx)
        opp_model: self-play 상대 (ActorCritic)
        device: torch device
        """
        if self.done:
            raise RuntimeError("Episode already finished")

        tile_slot = action // self.num_cells
        cell_idx = action % self.num_cells

        # 잘못된 액션 → 바로 패널티 주고 끝
        if tile_slot >= len(self.agent_hand) or self.board[cell_idx] != -1:
            return self._get_obs(), -5.0, True, False, {"invalid": True, "diff": self.prev_diff}

        # 1) 에이전트 수 적용
        self._apply_move(is_agent=True, slot=tile_slot, cell_idx=cell_idx)

        # 보드 꽉 차면 바로 종료
        if self.filled >= self.num_cells:
            return self._finalize_step()

        # 2) 상대 수 적용
        if opp_model is not None:
            # self-play 상대
            self._opponent_model_move(opp_model, device)
        else:
            # 내부 정책 (random/greedy/heuristic/mixed)
            self._opponent_policy_move()

        # 최종 상태 정리
        if self.filled >= self.num_cells:
            return self._finalize_step()
        return self._finalize_step(done_override=False)

    def _finalize_step(self, done_override: Optional[bool] = None):
        first_s, second_s, curr_diff = self._compute_scores()
        delta = curr_diff - self.prev_diff
        self.prev_diff = curr_diff
        self.last_delta = delta

        step_reward = float(np.clip(delta / self.reward_scale, -2.0, 2.0))

        is_done = (self.filled >= self.num_cells) if done_override is None else bool(done_override)
        self.done = is_done

        info = {
            "diff": float(curr_diff),
            "first": float(first_s),
            "second": float(second_s),
            "delta": float(delta),
        }

        if is_done:
            if curr_diff > 0:
                step_reward += self.win_bonus
            elif curr_diff < 0:
                step_reward -= self.win_bonus

        return self._get_obs(), step_reward, is_done, False, info

    def _apply_move(self, is_agent: bool, slot: int, cell_idx: int):
        hand = self.agent_hand if is_agent else self.opp_hand
        deck = self.agent_deck if is_agent else self.opp_deck

        if not hand:
            return None, None

        if slot >= len(hand):
            slot = 0

        tid = hand.pop(slot)
        self.board[cell_idx] = tid
        self.filled += 1

        drawn = None
        if deck:
            drawn = deck.pop()
            hand.append(drawn)

        return tid, drawn

    # ───────── opponent 관련 ─────────

    def _opponent_policy_move(self):
        """base_opponent_policy + greedy_prob + use_heuristic 조합."""
        if self.use_heuristic or self.base_opponent_policy == "heuristic":
            self._opponent_super_greedy_optimized()
        elif self.base_opponent_policy == "greedy":
            self._opponent_super_greedy_optimized()
        elif self.base_opponent_policy == "mixed":
            if self.rng.rand() < self.opponent_mix_greedy_prob:
                self._opponent_super_greedy_optimized()
            else:
                self._opponent_random()
        elif self.base_opponent_policy == "random":
            # 예전 greedy_prob와 섞어 사용
            if self.rng.rand() < self.greedy_prob:
                self._opponent_super_greedy_optimized()
            else:
                self._opponent_random()
        else:
            self._opponent_random()

    def _opponent_random(self):
        valid = [
            (s, c)
            for s in range(len(self.opp_hand))
            for c in range(self.num_cells)
            if self.board[c] == -1
        ]
        if not valid:
            return
        s, c = valid[self.rng.randint(len(valid))]
        self._apply_move(is_agent=False, slot=s, cell_idx=c)

    def _opponent_super_greedy_optimized(self):
        """
        강한 휴리스틱 상대:
          1) 공격 점수(색 연결) + 중앙 가중치로 후보 상위 k개 선정
          2) 그 후보들에 대해 간단한 방어 점수까지 고려
        """
        candidates = []  # (pre_score, slot, c_idx)

        for s, tid in enumerate(self.opp_hand):
            for c in range(self.num_cells):
                if self.board[c] != -1:
                    continue
                atk = self._local_score(c, tid, is_first=False)
                center_bonus = len(ADJ[c]) * 0.5
                candidates.append((atk + center_bonus, s, c))

        if not candidates:
            self._opponent_random()
            return

        # 상위 k개만 남기기 (속도 최적화)
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:5]

        best_val = -1e9
        best_moves = []

        for pre_score, s, c in top_k:
            defense = 0
            for p_tid in self.agent_hand:
                defense = max(defense, self._local_score(c, p_tid, is_first=True))

            final_val = pre_score + 0.8 * defense
            if final_val > best_val:
                best_val = final_val
                best_moves = [(s, c)]
            elif final_val == best_val:
                best_moves.append((s, c))

        if not best_moves:
            self._opponent_random()
            return

        s, c = best_moves[self.rng.randint(len(best_moves))]
        self._apply_move(is_agent=False, slot=s, cell_idx=c)

    def _opponent_model_move(self, model, device):
        """
        self-play opponent (PPO 모델)
        - obs는 opponent 관점으로 _get_obs(is_opponent=True)를 사용.
        """
        import torch

        obs = self._get_obs(is_opponent=True)
        mask = self.get_valid_action_mask(is_opponent=True)

        if not mask.any():
            self._opponent_random()
            return

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            # ActorCritic에 get_action 또는 forward 사용
            from torch.distributions import Categorical
            logits, _, _ = model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample().item()

        slot = action // self.num_cells
        c_idx = action % self.num_cells

        if slot < len(self.opp_hand) and self.board[c_idx] == -1:
            self._apply_move(is_agent=False, slot=slot, cell_idx=c_idx)
        else:
            self._opponent_random()

    # ───────── 스코어 계산 / 로컬 스코어 ─────────

    def _local_score(self, cell_idx: int, tid: int, is_first: bool) -> int:
        """
        해당 칸(cell_idx)에 tid 타일을 둘 때
        인접 타일과의 매칭으로 얻는 local score (제곱).
        - first: symbol 기반
        - second: color 기반
        """
        score = 1
        c_idx, s_idx = tid // 4, tid % 4

        for nb in ADJ[cell_idx]:
            nb_tid = self.board[nb]
            if nb_tid == -1:
                continue
            if is_first:
                match = (nb_tid % 4 == s_idx)
            else:
                match = (nb_tid // 4 == c_idx)
            if match:
                score += 1

        return score * score

    def _compute_scores(self) -> Tuple[int, int, int]:
        """
        (first_score, second_score, diff = first-second)
        """
        first = self._calc_uf(use_symbol=True)
        second = self._calc_uf(use_symbol=False)
        return first, second, first - second

    def _calc_uf(self, use_symbol: bool) -> int:
        """
        Connexion의 유니온-파인드 기반 점수 계산.
        - use_symbol=True  → symbol 기준 연결
        - use_symbol=False → color 기준 연결
        """
        parent = list(range(self.num_cells))
        size = [0] * self.num_cells

        for i, tid in enumerate(self.board):
            if tid != -1:
                size[i] = 1

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                size[ra] += size[rb]

        for i in range(self.num_cells):
            tid = self.board[i]
            if tid == -1:
                continue
            for nb in ADJ[i]:
                if nb > i:
                    continue
                ntid = self.board[nb]
                if ntid == -1:
                    continue
                if use_symbol:
                    match = (tid % 4 == ntid % 4)
                else:
                    match = (tid // 4 == ntid // 4)
                if match:
                    union(i, nb)

        score = 0
        for i in range(self.num_cells):
            if parent[i] == i and size[i] > 0:
                score += size[i] ** 2
        return score
