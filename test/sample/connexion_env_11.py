import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

# 외부 휴리스틱 엔진 제거 버전
heu_engine = None


# ─────────────────────────
# 기본 정의들
# ─────────────────────────
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


# 중앙 구멍들
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
            return (
                (dr == 0 and dc == 0)
                or (dr == 0 and dc == -1)
                or (dr == -1 and dc == 0)
            )
        if self.sign == Sign.PLUS and other.sign == Sign.MINUS:
            return (
                (dr == 0 and dc == 0)
                or (dr == 0 and dc == 1)
                or (dr == 1 and dc == 0)
            )
        return False


@dataclass(frozen=True)
class Tile:
    color: Color
    symbol: Symbol

    def __str__(self) -> str:
        return f"{self.color.value}{self.symbol.value}"


# ─────────────────────────
# 전역 테이블들
# ─────────────────────────
ALL_CELLS: List[Cell] = []
for col in Column:
    for row in Row:
        for sign in Sign:
            c = Cell(col, row, sign)
            if c.isValid():
                ALL_CELLS.append(c)

NUM_CELLS: int = len(ALL_CELLS)  # 64
CELL_INDEX: Dict[str, int] = {str(c): i for i, c in enumerate(ALL_CELLS)}

# 인접 리스트
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


# ─────────────────────────
# 환경 클래스
# ─────────────────────────
class ConnexionEnv:
    def __init__(
        self,
        seed: int = 0,
        my_role: int = 0,
        opponent_policy: str = "random",
    ):
        """
        my_role: 지금은 0(심볼 플레이어)만 사용한다고 보면 됨
        opponent_policy: "random", "mixed", "greedy" (PPO residual 코드에서 사용)
        """
        self.rng = np.random.RandomState(seed)
        self.num_cells = NUM_CELLS
        self.action_size = self.num_cells * 5  # 5장의 손패 * 64칸 = 320

        # Board(1024) + MyHand(80) + OppHand(80) = 1184
        self.obs_dim = self.num_cells * 16 + 5 * 16 + 5 * 16

        # 보상 관련
        self.reward_scale = 40.0
        self.win_bonus = 1.0

        # 간단한 greedy 제어용 (옛 PPO7 코드 호환)
        self.greedy_prob = 0.0

        # 휴리스틱(슈퍼 그리디) 사용 여부
        self.use_heuristic = False

        # 하이브리드 PPO용 상대 커리큘럼 설정
        self.my_role = my_role
        self.base_opponent_policy = opponent_policy  # "random" / "mixed" / "greedy"
        self.opponent_mix_greedy_prob = 0.5

        self.reset()

    # ───────── 외부 설정 메서드 ─────────
    def set_greedy_prob(self, prob: float):
        """옛 PPO7 코드에서 쓰던 greedy 비율 (fallback 용)"""
        self.greedy_prob = float(np.clip(prob, 0.0, 1.0))

    def set_heuristic_mode(self, use_heuristic: bool, is_first_for_heuristic: bool = False):
        """
        PPO7 코드 호환용.
        use_heuristic=True 이면 상대를 내부 super-greedy 휴리스틱으로 사용.
        is_first_for_heuristic는 현재 내부 로직에서는 사용하지 않음.
        """
        self.use_heuristic = use_heuristic

    # ───────── 내부 유틸 ─────────
    def _tid_to_str(self, tid: int) -> str:
        return COLOR_LIST[tid // 4].value + SYMBOL_LIST[tid % 4].value

    # ───────── reset ─────────
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        # 전체 덱 생성 (각 타일 4장씩 총 64장)
        full_deck_tiles = [
            Tile(c, s)
            for c in COLOR_LIST
            for s in SYMBOL_LIST
            for _ in range(4)
        ]
        self.rng.shuffle(full_deck_tiles)
        full_deck = [tile_to_id(t) for t in full_deck_tiles]

        # 27 + 5 분배 (에이전트/상대 대칭)
        self.agent_deck = full_deck[:27]
        self.opp_deck = full_deck[27:54]
        self.agent_hand = full_deck[54:59]
        self.opp_hand = full_deck[59:64]

        # 보드 상태
        self.board = [-1] * self.num_cells
        self.filled = 0
        self.done = False
        self.prev_diff = 0.0  # 이전 턴까지의 diff

        return self._get_obs(), {}

    # ───────── 관측 생성 ─────────
    def _get_obs(self, is_opponent: bool = False) -> np.ndarray:
        """
        is_opponent=True 이면 "상대 입장에서 본 관측"을 생성 (self-play 상대용)
        """
        my_target_hand = self.opp_hand if is_opponent else self.agent_hand
        opp_target_hand = self.agent_hand if is_opponent else self.opp_hand

        # 1) 보드 (64 * 16)
        board_feat = np.zeros((self.num_cells, 16), dtype=np.float32)
        for idx, tid in enumerate(self.board):
            if tid != -1:
                board_feat[idx, tid] = 1.0

        # 2) 내 손패 (5 * 16)
        my_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(my_target_hand):
            if slot < 5:
                my_hand_feat[slot, tid] = 1.0

        # 3) 상대 손패 (5 * 16)
        opp_hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(opp_target_hand):
            if slot < 5:
                opp_hand_feat[slot, tid] = 1.0

        return np.concatenate(
            [
                board_feat.flatten(),
                my_hand_feat.flatten(),
                opp_hand_feat.flatten(),
            ]
        )

    # ───────── 액션 마스크 ─────────
    def get_valid_action_mask(self, is_opponent: bool = False) -> np.ndarray:
        """
        is_opponent=True 이면 상대 관점의 valid mask.
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
        """
        PPO7에서 사용하던 "heuristic 후보 subset" 마스크.
        상위 점수 근처만 살아 있게 해서 exploration을 좀 더 똑똑하게.
        """
        valid_mask = self.get_valid_action_mask(is_opponent=False)
        if valid_mask.sum() == 0:
            return valid_mask

        mask = np.zeros(self.action_size, dtype=bool)
        scores = []
        indices = []

        for slot, tid in enumerate(self.agent_hand):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    idx = slot * self.num_cells + ci
                    s = self._local_score(ci, tid, is_first=True)
                    scores.append(s)
                    indices.append(idx)

        if not scores:
            return valid_mask

        max_s = max(scores)
        if max_s <= 0:
            # 전부 0이거나 음수면 그냥 full valid mask 사용
            return valid_mask

        threshold = max_s * 0.7
        count_smart = 0
        for s, idx in zip(scores, indices):
            if s >= threshold:
                mask[idx] = True
                count_smart += 1

        if count_smart == 0:
            return valid_mask
        return mask

    # ───────── PPO residual용 휴리스틱 logits ─────────
    def get_heuristic_logits(self) -> np.ndarray:
        """
        PPO residual에서 사용할 휴리스틱 logits.
        - 에이전트(심볼 플레이어) 입장에서의 지역 점수 + 중앙 가중치.
        - invalid action은 0.0 (어차피 마스크에서 막힘)
        """
        logits = np.zeros(self.action_size, dtype=np.float32)
        if self.done:
            return logits

        for slot, tid in enumerate(self.agent_hand):
            for c in range(self.num_cells):
                if self.board[c] != -1:
                    continue
                idx = slot * self.num_cells + c

                base = self._local_score(c, tid, is_first=True)
                center_bonus = len(ADJ[c]) * 0.3
                logits[idx] = base + center_bonus

        return logits

    # ───────── PPO delta-head용 delta 추정 ─────────
    def estimate_my_delta_for_action(self, action_idx: int) -> float:
        """
        현재 diff 기준으로, '이 액션만' 했을 때 diff가 얼마나 바뀌는지 근사.
        - 상대는 아직 두지 않았다고 가정.
        """
        if self.done:
            return 0.0

        slot = action_idx // self.num_cells
        c_idx = action_idx % self.num_cells

        if slot >= len(self.agent_hand) or self.board[c_idx] != -1:
            return -100.0  # 명백한 invalid이면 큰 음수

        # --- 상태 백업 ---
        saved_board = list(self.board)
        saved_filled = self.filled
        saved_agent_hand = list(self.agent_hand)
        saved_agent_deck = list(self.agent_deck)
        saved_opp_hand = list(self.opp_hand)
        saved_opp_deck = list(self.opp_deck)
        saved_prev_diff = self.prev_diff
        saved_done = self.done

        # --- 가상으로 한 수 둬보기 ---
        tid = self.agent_hand.pop(slot)
        self.board[c_idx] = tid
        self.filled += 1

        if self.agent_deck:
            drawn = self.agent_deck.pop()
            self.agent_hand.append(drawn)

        _, _, new_diff = self._compute_scores()
        delta = new_diff - saved_prev_diff

        # --- 되돌리기 ---
        self.board = saved_board
        self.filled = saved_filled
        self.agent_hand = saved_agent_hand
        self.agent_deck = saved_agent_deck
        self.opp_hand = saved_opp_hand
        self.opp_deck = saved_opp_deck
        self.prev_diff = saved_prev_diff
        self.done = saved_done

        return float(delta)

    # ───────── step ─────────
    def step(self, action: int, opp_model=None, device=None):
        """
        opp_model/device는 PPO7 self-play에서 사용 (optional).
        PPO residual 코드에서는 action만 넣어서 호출.
        """
        if self.done:
            raise RuntimeError("Episode done")

        tile_slot = action // self.num_cells
        cell_idx = action % self.num_cells

        # invalid action 패널티
        if tile_slot >= len(self.agent_hand) or self.board[cell_idx] != -1:
            return self._get_obs(), -10.0, True, False, {"invalid": True}

        # 에이전트 수 적용
        self._apply_move(is_agent=True, slot=tile_slot, cell_idx=cell_idx)
        if self.filled >= self.num_cells:
            return self._finalize_step()

        # 상대 차례
        if opp_model is not None and not self.use_heuristic:
            self._opponent_model_move(opp_model, device)
        elif self.use_heuristic:
            # 내부 super-greedy 휴리스틱
            self._opponent_super_greedy_optimized()
        else:
            self._opponent_move()

        if self.filled >= self.num_cells:
            return self._finalize_step()
        return self._finalize_step(done_override=False)

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

        return self._get_obs(), step_reward, is_done, False, info

    # ───────── 수 적용 ─────────
    def _apply_move(self, is_agent: bool, slot: int, cell_idx: int):
        hand = self.agent_hand if is_agent else self.opp_hand
        deck = self.agent_deck if is_agent else self.opp_deck

        if slot >= len(hand):
            if not hand:
                return None, None
            slot = 0

        tid = hand.pop(slot)
        self.board[cell_idx] = tid
        self.filled += 1

        drawn = None
        if deck:
            drawn = deck.pop()
            hand.append(drawn)

        return tid, drawn

    # ───────── 최적화된 super-greedy 휴리스틱 ─────────
    def _opponent_super_greedy_optimized(self):
        """
        공격 점수 + 중앙 가중치로 상위 5 후보만 뽑고,
        거기에 방어 점수까지 더해서 최종 수 선택.
        """
        candidates = []  # (atk+center, slot, cell_idx)

        for s, tid in enumerate(self.opp_hand):
            for c in range(self.num_cells):
                if self.board[c] == -1:
                    atk = self._local_score(c, tid, is_first=False)
                    center_bonus = len(ADJ[c]) * 0.5
                    candidates.append((atk + center_bonus, s, c))

        if not candidates:
            self._opponent_random()
            return

        # 상위 5개만 사용 (pruning)
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:5]

        best_val = -9999.0
        best_moves: List[Tuple[int, int]] = []

        for pre_score, s, c in top_k:
            defense = 0.0
            # 플레이어(심볼)가 이 자리에 뒀을 때 얻는 최대 local score
            for p_tid in self.agent_hand:
                defense = max(defense, self._local_score(c, p_tid, is_first=True))

            final_val = pre_score + defense * 0.8

            if final_val > best_val:
                best_val = final_val
                best_moves = [(s, c)]
            elif final_val == best_val:
                best_moves.append((s, c))

        if best_moves:
            s, c = best_moves[self.rng.randint(len(best_moves))]
            self._apply_move(False, s, c)
        else:
            self._opponent_random()

    # ───────── 모델 기반 상대(self-play best) ─────────
    def _opponent_model_move(self, model, device):
        import torch

        obs = self._get_obs(is_opponent=True)
        mask = self.get_valid_action_mask(is_opponent=True)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            ppo_idx = torch.argmax(logits, dim=1).item()

        slot = ppo_idx // self.num_cells
        c_idx = ppo_idx % self.num_cells

        if slot < len(self.opp_hand) and self.board[c_idx] == -1:
            self._apply_move(False, slot, c_idx)
        else:
            self._opponent_random()

    # ───────── 기본 상대 정책 ─────────
    def _opponent_move(self):
        """
        base_opponent_policy가 설정되어 있으면 그것 우선.
        아니면 예전 greedy_prob 로직 사용 (PPO7 호환).
        """
        policy = getattr(self, "base_opponent_policy", None)

        if policy == "random":
            self._opponent_random()

        elif policy == "greedy":
            self._opponent_super_greedy_optimized()

        elif policy == "mixed":
            p = getattr(self, "opponent_mix_greedy_prob", 0.5)
            if self.rng.rand() < p:
                self._opponent_super_greedy_optimized()
            else:
                self._opponent_random()

        else:
            # fallback: 옛 방식
            if self.rng.rand() < self.greedy_prob:
                self._opponent_super_greedy_optimized()
            else:
                self._opponent_random()

    def _opponent_random(self):
        valid: List[Tuple[int, int]] = []
        for s in range(len(self.opp_hand)):
            for c in range(self.num_cells):
                if self.board[c] == -1:
                    valid.append((s, c))

        if valid:
            s, c = valid[self.rng.randint(len(valid))]
            self._apply_move(False, s, c)

    # ───────── 점수 관련 ─────────
    def _local_score(self, cell_idx: int, tid: int, is_first: bool) -> int:
        """
        특정 칸에 타일 하나를 놓았을 때, 그 칸 주변 연결로 인한 지역 점수.
        first(심볼) / second(컬러) 여부에 따라 기준이 달라짐.
        """
        score = 1
        c_idx = tid // 4  # 색 index
        s_idx = tid % 4   # 심볼 index

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
        (first_score, second_score, diff) 반환
        first: 심볼 기준 그룹 점수
        second: 색 기준 그룹 점수
        """
        first = self._calc_uf(use_symbol=True)
        second = self._calc_uf(use_symbol=False)
        return first, second, first - second

    def _calc_uf(self, use_symbol: bool) -> int:
        """
        UF로 연결 컴포넌트 크기^2 합 계산.
        use_symbol=True -> 심볼 기준
        use_symbol=False -> 색 기준
        """
        parent = list(range(self.num_cells))
        size = [0] * self.num_cells

        for i, tid in enumerate(self.board):
            if tid != -1:
                size[i] = 1

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
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
