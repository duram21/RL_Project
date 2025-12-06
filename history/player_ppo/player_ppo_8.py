import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import torch


class Column(Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"


class Row(Enum):
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"
    _5 = "5"
    _6 = "6"


class Sign(Enum):
    MINUS = "-"
    PLUS = "+"


class Color(Enum):
    R = "R"
    G = "G"
    B = "B"
    Y = "Y"


class Symbol(Enum):
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"


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
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        if self.sign == Sign.MINUS and other.sign == Sign.PLUS:
            return (dr == dc == 0) or (dr == 0 and dc == -1) or (dr == -1 and dc == 0)
        if self.sign == Sign.PLUS and other.sign == Sign.MINUS:
            return (dr == dc == 0) or (dr == 0 and dc == 1) or (dr == 1 and dc == 0)
        return False


@dataclass(frozen=True)
class Tile:
    color: Color
    symbol: Symbol

    def __str__(self) -> str:
        return f"{self.color.value}{self.symbol.value}"


# ----- 보드 / 타일 전역 유틸 -----

ALL_CELLS: List[Cell] = []
for col in Column:
    for row in Row:
        for sign in Sign:
            c = Cell(col, row, sign)
            if c.isValid():
                ALL_CELLS.append(c)

NUM_CELLS: int = len(ALL_CELLS)
CELL_INDEX: Dict[str, int] = {str(c): i for i, c in enumerate(ALL_CELLS)}

ADJACENT: List[List[int]] = [[] for _ in range(NUM_CELLS)]
for i, ci in enumerate(ALL_CELLS):
    for j, cj in enumerate(ALL_CELLS):
        if i != j and ci.isAdjacent(cj):
            ADJACENT[i].append(j)

COLOR_LIST = [Color.R, Color.G, Color.B, Color.Y]
SYMBOL_LIST = [Symbol._1, Symbol._2, Symbol._3, Symbol._4]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LIST)}
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOL_LIST)}


def tile_to_id(t: Tile) -> int:
    return COLOR_TO_IDX[t.color] * 4 + SYMBOL_TO_IDX[t.symbol]


def id_to_tile(tid: int) -> Tile:
    return Tile(COLOR_LIST[tid // 4], SYMBOL_LIST[tid % 4])


class ConnexionEnv:
    """
    - 선공(에이전트): 문양 기준 점수
    - 후공(상대): 색 기준 점수

    관측(obs):
      - 보드: 64칸 × 16채널 one-hot → 1024
      - 내 손패: 5장 × 16 → 80
      - 상대 손패: 5장 × 16 → 80
      => obs_dim = 1184
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

        self.num_cells = NUM_CELLS
        self.action_size = self.num_cells * 5

        # Board(1024) + MyHand(80) + OppHand(80)
        self.obs_dim = self.num_cells * 16 + 5 * 16 + 5 * 16

        # 보상 관련
        self.reward_scale = 40.0
        self.win_bonus = 1.0

        # 규칙 기반 상대의 "탐욕 정도"
        self.greedy_prob = 0.0

        self.reset()

    # ----------------- 난이도 / 시드 -----------------

    def set_greedy_prob(self, prob: float):
        """규칙 기반 상대의 greedy 사용 비율(0~1)"""
        self.greedy_prob = float(np.clip(prob, 0.0, 1.0))

    # ----------------- 초기화 / 관측 -----------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        # 16 종류 × 4장 = 64장 전체 덱
        full_deck_tiles = [Tile(c, s) for c in COLOR_LIST for s in SYMBOL_LIST for _ in range(4)]
        self.rng.shuffle(full_deck_tiles)
        full_deck = [tile_to_id(t) for t in full_deck_tiles]

        # 덱/손패 분배 (네가 쓰던 방식 유지)
        self.agent_deck = full_deck[:27]
        self.opp_deck = full_deck[27:54]
        self.agent_hand = full_deck[54:59]
        self.opp_hand = full_deck[59:64]

        self.board = [-1] * self.num_cells
        self.filled = 0
        self.done = False
        self.prev_diff = 0.0

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def _get_obs(self, is_opponent: bool = False) -> np.ndarray:
        """
        is_opponent=False → 선공(에이전트) 기준 관측
        is_opponent=True  → 후공(상대) 기준 관측
        """
        my_hand = self.opp_hand if is_opponent else self.agent_hand
        opp_hand = self.agent_hand if is_opponent else self.opp_hand

        # 1) 보드
        board_feat = np.zeros((self.num_cells, 16), dtype=np.float32)
        for idx, tid in enumerate(self.board):
            if tid != -1:
                board_feat[idx, tid] = 1.0

        # 2) 내 손패
        my_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(my_hand):
            if slot >= 5:
                break
            my_hand_feat[slot, tid] = 1.0

        # 3) 상대 손패
        opp_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(opp_hand):
            if slot >= 5:
                break
            opp_hand_feat[slot, tid] = 1.0

        obs = np.concatenate(
            [board_feat.flatten(), my_hand_feat.flatten(), opp_hand_feat.flatten()]
        )
        return obs

    # ----------------- 액션 마스크 -----------------

    def get_valid_action_mask(self, is_opponent: bool = False) -> np.ndarray:
        """
        규칙상 가능한 (hand_slot, cell_idx)에 해당하는 것만 True 인 마스크
        """
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done:
            return mask

        hand = self.opp_hand if is_opponent else self.agent_hand
        for slot in range(len(hand)):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    mask[slot * self.num_cells + ci] = True
        return mask

    def get_smart_action_mask(self, is_opponent: bool = False) -> np.ndarray:
        """
        규칙상 가능한 수 중에서,
        - 선공: 문양 기준 로컬 그룹이 큰 수
        - 후공: 색 기준 로컬 그룹이 큰 수
        만 남기는 "똑똑한" 마스크.
        """
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done:
            return mask

        hand = self.opp_hand if is_opponent else self.agent_hand
        is_first = not is_opponent  # 선공=문양, 후공=색

        scores: List[float] = []
        indices: List[int] = []

        for slot, tid in enumerate(hand):
            for ci in range(self.num_cells):
                if self.board[ci] != -1:
                    continue
                idx = slot * self.num_cells + ci
                s = self._local_score(ci, tid, is_first=is_first)
                scores.append(s)
                indices.append(idx)

        if not scores:
            return mask

        max_s = max(scores)

        # 전부 0이면 "좋은 수"가 구분 안 되므로 가능한 수 전부 허용
        if max_s <= 0:
            for idx in indices:
                mask[idx] = True
            return mask

        threshold = max_s * 0.7
        for s, idx in zip(scores, indices):
            if s >= threshold:
                mask[idx] = True

        # 혹시 threshold 때문에 아무것도 안 남으면 전부 허용
        if not mask.any():
            for idx in indices:
                mask[idx] = True

        return mask

    # ----------------- step -----------------

    def step(self, action: int, opp_model=None, device=None):
        """
        action: 에이전트(선공)가 둘 액션 인덱스
        opp_model: 후공이 best.pt 같은 모델을 쓸 때 넘겨주는 ActorCritic (선택)
        device: PyTorch device
        """
        if self.done:
            raise RuntimeError("Episode done; call reset().")

        tile_slot = action // self.num_cells
        cell_idx = action % self.num_cells

        # 방어적 체크 (학습에선 마스크가 막아줘야 정상)
        if (
            tile_slot < 0
            or tile_slot >= len(self.agent_hand)
            or cell_idx < 0
            or cell_idx >= self.num_cells
            or self.board[cell_idx] != -1
        ):
            self.done = True
            return self._get_obs(), -10.0, True, False, {"invalid": True}

        # 1) 에이전트 수 두기
        self._apply_move(is_agent=True, slot=tile_slot, cell_idx=cell_idx)

        # 보드가 꽉 찼다면 후공은 둘 수 없음 → 바로 마무리
        if self.filled >= self.num_cells:
            return self._finalize_step()

        # 2) 상대 수 두기 (모델 or 규칙 기반)
        if opp_model is not None:
            self._opponent_model_move(opp_model, device)
        else:
            self._opponent_move()

        # 3) 최종 보상 계산
        if self.filled >= self.num_cells:
            return self._finalize_step()
        else:
            return self._finalize_step(done_override=False)

    # ----------------- 내부 유틸 -----------------

    def _finalize_step(self, done_override: Optional[bool] = None):
        first_s, second_s, curr_diff = self._compute_scores()
        delta = curr_diff - self.prev_diff
        self.prev_diff = curr_diff

        step_reward = float(np.clip(delta / self.reward_scale, -2.0, 2.0))

        is_done = (self.filled >= self.num_cells) if done_override is None else done_override
        self.done = is_done

        info = {"diff": curr_diff, "first": first_s, "second": second_s}

        if is_done:
            if curr_diff > 0:
                step_reward += self.win_bonus
            elif curr_diff < 0:
                step_reward -= self.win_bonus

        obs = self._get_obs()
        return obs, step_reward, is_done, False, info

    def _apply_move(self, is_agent: bool, slot: int, cell_idx: int):
        hand = self.agent_hand if is_agent else self.opp_hand
        deck = self.agent_deck if is_agent else self.opp_deck

        if slot < 0 or slot >= len(hand):
            # 방어적: 잘못된 slot이면 아무 것도 안 함
            return

        tid = hand.pop(slot)
        self.board[cell_idx] = tid
        self.filled += 1

        if deck:
            hand.append(deck.pop())

    # ----------------- 모델 기반 상대 수 (best.pt) -----------------

    def _opponent_model_move(
        self,
        model,
        device,
        lambda_gain: float = 0.7,
        top_k: int = 32,
    ):
        """
        best.pt 같은 PPO 정책을 사용하는 '강한 상대' 한 수 두기.
        - Smart Mask(후공 기준)로 후보를 줄인 뒤
        - PPO 확률 + 즉시 gain(s2 - s1)를 합쳐 점수가 가장 큰 수 선택
        """
        if not self.opp_hand:
            return

        # 1) 관측 & 마스크
        obs = self._get_obs(is_opponent=True)

        valid_mask = self.get_valid_action_mask(is_opponent=True)
        if not valid_mask.any():
            return

        smart_mask = self.get_smart_action_mask(is_opponent=True)
        cand_mask = np.logical_and(valid_mask, smart_mask)
        if not cand_mask.any():
            cand_mask = valid_mask

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            # ActorCritic.forward(obs) → (logits, value) 라고 가정
            logits, _ = model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        cand_indices = np.where(cand_mask)[0]
        if cand_indices.size == 0:
            self._opponent_random()
            return

        # 2) 정책 확률 기준 상위 top_k만 남기기 (속도 절약)
        if top_k is not None and cand_indices.size > top_k:
            cand_probs = probs[cand_indices]
            order = np.argsort(-cand_probs)
            cand_indices = cand_indices[order[:top_k]]

        # 3) 후보마다 즉시 gain 계산 (색 기준 s2 - s1)
        gains = []
        for idx in cand_indices:
            g = self._simulate_opp_score(idx)
            gains.append(g)
        gains = np.array(gains, dtype=np.float32)

        # gains 정규화 (0~1)
        if gains.size > 0 and gains.max() > gains.min():
            gains_norm = (gains - gains.min()) / (gains.max() - gains.min() + 1e-8)
        else:
            gains_norm = np.zeros_like(gains)

        probs_cand = np.clip(probs[cand_indices], 1e-8, 1.0)
        log_prob = np.log(probs_cand)

        # 최종 점수 = log π(a|s) + λ · gains_norm
        final_scores = log_prob + float(lambda_gain) * gains_norm
        best_idx = int(final_scores.argmax())
        best_global_idx = int(cand_indices[best_idx])

        slot = best_global_idx // self.num_cells
        c_idx = best_global_idx % self.num_cells

        if 0 <= slot < len(self.opp_hand) and self.board[c_idx] == -1:
            self._apply_move(is_agent=False, slot=slot, cell_idx=c_idx)
        else:
            # 혹시나 이상할 경우 랜덤으로 fallback
            self._opponent_random()

    def _simulate_opp_score(self, action_idx: int) -> float:
        """
        후공(색 기준)이 action_idx에 둔다고 가정했을 때,
        즉시 gain = (색 점수 - 문양 점수)를 계산.
        """
        slot = action_idx // self.num_cells
        c_idx = action_idx % self.num_cells

        if slot < 0 or slot >= len(self.opp_hand) or self.board[c_idx] != -1:
            return -1e9  # 완전 잘못된 수는 최악으로

        bak_hand = list(self.opp_hand)
        bak_board = list(self.board)
        bak_filled = self.filled

        tid = self.opp_hand.pop(slot)
        self.board[c_idx] = tid
        self.filled += 1

        s1, s2, diff = self._compute_scores()
        gain = float(s2 - s1)

        # 원상복구
        self.board = bak_board
        self.opp_hand = bak_hand
        self.filled = bak_filled

        return gain

    # ----------------- 규칙 기반 상대 수 (랜덤/그리디) -----------------

    def _opponent_move(self):
        if not self.opp_hand:
            return
        if self.rng.rand() < self.greedy_prob:
            self._opponent_sample_greedy()
        else:
            self._opponent_random()

    def _opponent_random(self):
        valid: List[Tuple[int, int]] = []
        for s in range(len(self.opp_hand)):
            for c in range(self.num_cells):
                if self.board[c] == -1:
                    valid.append((s, c))
        if not valid:
            return
        s, c = valid[self.rng.randint(len(valid))]
        self._apply_move(False, s, c)

    def _opponent_sample_greedy(self):
        best_val = -1e9
        best_moves: List[Tuple[int, int]] = []

        for s, tid in enumerate(self.opp_hand):
            for c in range(self.num_cells):
                if self.board[c] != -1:
                    continue
                val = self._local_score(c, tid, is_first=False)  # 후공: 색 기준
                if val > best_val:
                    best_val = val
                    best_moves = [(s, c)]
                elif val == best_val:
                    best_moves.append((s, c))

        if not best_moves:
            return
        s, c = best_moves[self.rng.randint(len(best_moves))]
        self._apply_move(False, s, c)

    # ----------------- 점수 계산 -----------------

    def _local_score(self, cell_idx: int, tid: int, is_first: bool) -> int:
        """
        인접한 같은 속성 타일 개수 기반 로컬 점수 (크기^2).
        is_first=True  → 문양 기준 (선공)
        is_first=False → 색 기준  (후공)
        """
        score = 1
        c_idx, s_idx = tid // 4, tid % 4

        for nb in ADJACENT[cell_idx]:
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
        현재 보드에서:
        - first_score  (문양 기준)
        - second_score (색 기준)
        - diff = first_score - second_score
        """
        first_score = self._calc_uf(use_symbol=True)
        second_score = self._calc_uf(use_symbol=False)
        diff = first_score - second_score
        return first_score, second_score, diff

    def _calc_uf(self, use_symbol: bool) -> int:
        """
        UF(Union-Find)로 같은 속성끼리 연결된 컴포넌트의 (크기^2) 합을 계산.
        use_symbol=True  → 문양 기준
        use_symbol=False → 색 기준
        """
        parent = list(range(self.num_cells))
        size = [0] * self.num_cells

        def find(x: int) -> int:
            # path compression
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            # union by size
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        # 초기 사이즈 설정
        for i, tid in enumerate(self.board):
            if tid != -1:
                size[i] = 1

        # 인접한 칸끼리 속성이 같으면 union
        for i, tid in enumerate(self.board):
            if tid == -1:
                continue
            for nb in ADJACENT[i]:
                if nb > i:
                    continue
                nb_tid = self.board[nb]
                if nb_tid == -1:
                    continue

                if use_symbol:
                    if (tid % 4) != (nb_tid % 4):
                        continue
                else:
                    if (tid // 4) != (nb_tid // 4):
                        continue

                union(i, nb)

        score = 0
        for i in range(self.num_cells):
            if parent[i] == i and size[i] > 0:
                score += size[i] * size[i]
        return score
