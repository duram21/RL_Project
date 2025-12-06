# nypc_easy_bot.py
# (멍청한 버전: 수읽기 폭 감소, 방어 무시, 중앙 지향 제거)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import sys, os, struct, time
from enum import Enum

# ───────────────────── 파라미터 (멍청한 버전) ─────────────────────
# 상대 득점항 억제(λ), 상대 응수 민감도(μ) -> 대폭 감소 (상대 신경 안 씀)
LAMBDA_BASE_FIRST  = 0.1
LAMBDA_BASE_SECOND = 0.1
MU_BASE_FIRST      = 0.0
MU_BASE_SECOND     = 0.0

NU_DEF      = 0.0

# 탐색 폭/깊이 -> 대폭 축소 (거의 1개만 봄)
K_MY_DEF    = 4   # 기존 28 -> 4 (내 후보 4개만 봄)
K_OPP_DEF   = 2   # 기존 18 -> 2 (상대 응수 2개만 봄)
K_RE_DEF    = 1   # 기존 12 -> 1
K_OPP2_DEF  = 1   # 기존 12 -> 1

THRESH_Q1   = 100 # 퀼리티 문턱을 높여서 깊게 안 들어가게 함
THRESH_Q2   = 100
GAMMA_Q     = 0.1

USE_CELL_BIAS = False # 위치 가중치 끔
CELL_BIAS_DEFAULT = [0.0]*64

# 손패/자유도 보너스 -> 제거
ALPHA_SYNERGY = 0.0
BETA_LIBERTY  = 0.0

# ── 중앙/차수 가중치 -> 0.0으로 설정 (아무 데나 둠)
CENTER_W_EARLY_FIRST = 0.0
CENTER_W_MID_FIRST   = 0.0
CENTER_W_LATE_FIRST  = 0.0

CENTER_W_EARLY_SECOND= 0.0
CENTER_W_MID_SECOND  = 0.0
CENTER_W_LATE_SECOND = 0.0

CENTER2_W_EARLY = 0.0
CENTER2_W_MID   = 0.0
CENTER2_W_LATE  = 0.0

DEG_W         = 0.0

# 블록 보상 -> 제거 (방어 안 함)
BLOCK_W       = 0.0

# 위협 증가(Δ) 패널티 -> 제거 (상대가 세지는 거 신경 안 씀)
THREAT_DELTA_W_FIRST  = 0.0
THREAT_DELTA_W_SECOND = 0.0

# 로컬 핫스팟 패널티 -> 제거
RISK_LOCAL_W_EARLY = 0.0
RISK_LOCAL_W_MID   = 0.0
RISK_LOCAL_W_LATE  = 0.0
MAX_LOCAL_CHECKS   = 1

# 인접 패널티 -> 제거 (상대 도와주는 수 안 피함)
ETA_MATCH_COL = 0.0
ETA_MATCH_SYM = 0.0

ISO_COL_W_FIRST     = 0.0
EARLY_EDGE_PENALTY  = 0.0  # 구석에도 막 둠

# Must-Block guard -> 제거
MB_THRESH_ABS   = 999
MB_THRESH_RATIO = 1.0
MB_BONUS_MUL    = 0.0
MB_PENALTY_MUL  = 0.0

# ───────────────────── 보드/타일 (이하 로직 동일) ─────────────────────
class Column(Enum): A="a"; B="b"; C="c"; D="d"; E="e"; F="f"
class Row(Enum):    _1="1"; _2="2"; _3="3"; _4="4"; _5="5"; _6="6"
class Sign(Enum):   MINUS="-"; PLUS="+"
class Color(Enum):  R="R"; G="G"; B="B"; Y="Y"
class Symbol(Enum): _1="1"; _2="2"; _3="3"; _4="4"

FORBIDDEN = {"a1-","a4-","c3+","c6+","d1-","d4-","f3+","f6+"}

@dataclass(frozen=True)
class Cell:
    col: Column; row: Row; sign: Sign
    def __str__(self)->str: return f"{self.col.value}{self.row.value}{self.sign.value}"
    def is_valid(self)->bool: return str(self) not in FORBIDDEN
    def is_adjacent(self, other:"Cell")->bool:
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        if self.sign==Sign.MINUS and other.sign==Sign.PLUS:
            return (dr==0 and dc==0) or (dr==0 and dc==-1) or (dr==-1 and dc==0)
        if self.sign==Sign.PLUS and other.sign==Sign.MINUS:
            return (dr==0 and dc==0) or (dr==0 and dc== 1) or (dr== 1 and dc==0)
        return False
    @staticmethod
    def all_cells()->List["Cell"]:
        out=[]
        for c in Column:
            for r in Row:
                for s in Sign:
                    cell=Cell(c,r,s)
                    if cell.is_valid(): out.append(cell)
        return out
    @staticmethod
    def from_string(s:str)->"Cell":
        assert len(s)==3
        cell = Cell(Column(s[0]), Row(s[1]), Sign(s[2]))
        assert cell.is_valid()
        return cell

@dataclass(frozen=True)
class Tile:
    color: Color; symbol: Symbol
    def __str__(self)->str: return f"{self.color.value}{self.symbol.value}"
    @staticmethod
    def from_string(s:str)->"Tile":
        assert len(s)==2
        return Tile(Color(s[0]), Symbol(s[1]))

ALL_CELLS: List[Cell] = Cell.all_cells()
V = len(ALL_CELLS)
KEY2IDX: Dict[str,int] = {str(ALL_CELLS[i]): i for i in range(V)}

# 인접/중앙성/차수
ADJ: List[Tuple[int,...]] = [tuple() for _ in range(V)]
_tmp=[[] for _ in range(V)]
for i,a in enumerate(ALL_CELLS):
    for j,b in enumerate(ALL_CELLS):
        if i!=j and a.is_adjacent(b): _tmp[i].append(j)
ADJ=[tuple(x) for x in _tmp]
DEG=[len(ADJ[i]) for i in range(V)]

def center_score(i:int)->float:
    key=str(ALL_CELLS[i]); c=ord(key[0])-ord('a'); r=int(key[1])-1
    # 맨해튼 중심성(음수: 0이 중심)
    return - (abs(c-2.5)+abs(r-2.5))

def center_score2(i:int)->float:
    # 제곱(유클리드 유사) 중심성(더 강한 중앙 선호)
    key=str(ALL_CELLS[i]); c=ord(key[0])-ord('a'); r=int(key[1])-1
    dx=(c-2.5); dy=(r-2.5)
    return - (dx*dx + dy*dy)

CENTER=[center_score(i) for i in range(V)]
CENTER2=[center_score2(i) for i in range(V)]

# 타일 ID 변환
COLOR2IDX = {Color.R:0, Color.G:1, Color.B:2, Color.Y:3}
SYMB2IDX  = {Symbol._1:0, Symbol._2:1, Symbol._3:2, Symbol._4:3}
def tile_id(t:Tile)->int: return COLOR2IDX[t.color]*4 + SYMB2IDX[t.symbol]
def color_of(tid:int)->int:  return tid//4
def symbol_of(tid:int)->int: return tid%4

# ───────────────────── DSU + Undo ─────────────────────
class DSU:
    __slots__=("p","sz","sum_sq","n","hist")
    def __init__(self,n:int):
        self.n=n
        self.p=list(range(n))
        self.sz=[0]*n
        self.sum_sq=0
        self.hist=[]
    def find(self,x:int)->int:
        while self.p[x]!=x: x=self.p[x]
        return x
    def add_single(self,x:int):
        if self.sz[x]==0:
            self.hist.append(('a',x,self.sum_sq))
            self.p[x]=x; self.sz[x]=1; self.sum_sq+=1
    def unite(self,a:int,b:int):
        a=self.find(a); b=self.find(b)
        if a==b: return
        if self.sz[a]<self.sz[b]: a,b=b,a
        A=self.sz[a]; B=self.sz[b]; prev=self.sum_sq
        self.sum_sq += (A+B)*(A+B) - A*A - B*B
        self.p[b]=a; self.sz[a]=A+B
        self.hist.append(('u', b, a, A, B, prev))
    def snapshot(self)->int: return len(self.hist)
    def rollback(self, snap:int):
        while len(self.hist)>snap:
            rec=self.hist.pop()
            if rec[0]=='a':
                _,x,prev=rec; self.sz[x]=0; self.sum_sq=prev
            else:
                _,b_root,a_root,A,B,prev=rec
                self.p[b_root]=b_root; self.sz[a_root]=A; self.sum_sq=prev

def make_uf8()->List[DSU]: return [DSU(V) for _ in range(8)]
def delta_attr(occ:List[int], dsu:DSU, v:int, is_same_pred)->int:
    seen=set(); sum_sz=0; sum_sq=0
    for nb in ADJ[v]:
        tid=occ[nb]
        if tid==-1: continue
        if not is_same_pred(tid): continue
        r=dsu.find(nb)
        if r in seen: continue
        s=dsu.sz[r]
        sum_sz+=s; sum_sq+=s*s
    S=1+sum_sz
    return S*S - sum_sq

# data.bin 로딩 (있어도 멍청한 파라미터로 덮어씌우므로 무시)
def load_data_bin():
    # 강제로 멍청한 값 리턴
    return (0.1, 0.1), [0.0]*64

class TimeGuard:
    def __init__(self,budget_ms:int):
        self.start=time.perf_counter()
        self.budget=budget_ms/1000.0
    def over(self)->bool:
        return (time.perf_counter()-self.start)>self.budget

def multiset_minus_one(ids: List[int], x: int) -> List[int]:
    removed=False; out=[]
    for v in ids:
        if not removed and v==x:
            removed=True; continue
        out.append(v)
    return out

# ───────────────────── 엔진 ─────────────────────
class Game:
    def __init__(self, myTiles:List[Tile], oppTiles:List[Tile], isFirst:bool):
        self.isFirst=isFirst
        self.myTiles=list(myTiles)
        self.oppTiles=list(oppTiles)
        self.occ=[-1]*V
        self.uf8=make_uf8()
        self.occ_hist=[]
        # 파라미터 강제 할당
        self.lambda_base_first  = LAMBDA_BASE_FIRST
        self.lambda_base_second = LAMBDA_BASE_SECOND
        self.cellBias = CELL_BIAS_DEFAULT[:]

    # 후보 칸(프런티어 + 중앙 일부 강화)
    def frontier_empties(self)->List[int]:
        occ=self.occ; empties=[]; has_any=False
        for i in range(V):
            if occ[i]!=-1:
                has_any=True
                for nb in ADJ[i]:
                    if occ[nb]==-1: empties.append(nb)
        if not has_any:
            idxs=[i for i in range(V) if occ[i]==-1]
            idxs.sort(key=lambda x: (CENTER[x]+0.3*DEG[x] + 0.2*CENTER2[x]), reverse=True)
            return idxs[:36]
        if not empties:
            return [i for i in range(V) if occ[i]==-1]
        extras = sorted([i for i in range(V) if occ[i]==-1],
                        key=lambda x: CENTER[x]+0.3*DEG[x]+0.2*CENTER2[x], reverse=True)[:16]
        s=set(empties); s.update(extras)
        return list(s)

    # 적용/스냅/롤백
    def apply(self, v:int, tid:int):
        self.occ_hist.append((v,tid)); self.occ[v]=tid
        ci=color_of(tid); si=4+symbol_of(tid)
        self.uf8[ci].add_single(v)
        for nb in ADJ[v]:
            t=self.occ[nb]
            if t!=-1 and color_of(t)==ci: self.uf8[ci].unite(v, nb)
        self.uf8[si].add_single(v)
        for nb in ADJ[v]:
            t=self.occ[nb]
            if t!=-1 and (4+symbol_of(t))==si: self.uf8[si].unite(v, nb)

    def snapshot(self):
        return (len(self.occ_hist),
                self.uf8[0].snapshot(), self.uf8[1].snapshot(), self.uf8[2].snapshot(), self.uf8[3].snapshot(),
                self.uf8[4].snapshot(), self.uf8[5].snapshot(), self.uf8[6].snapshot(), self.uf8[7].snapshot())

    def rollback(self, snap):
        occ_len=snap[0]
        while len(self.occ_hist)>occ_len:
            v,tid=self.occ_hist.pop()
            self.occ[v]=-1
        for i in range(8):
            self.uf8[i].rollback(snap[1+i])

    # Δ(내/상대)
    def eval_delta_both(self, v:int, tid:int)->Tuple[int,int]:
        me_first=self.isFirst
        if me_first:
            d_me  = delta_attr(self.occ, self.uf8[4+symbol_of(tid)], v, lambda nb_tid,tid=tid: symbol_of(nb_tid)==symbol_of(tid))
            d_opp = delta_attr(self.occ, self.uf8[color_of(tid)],       v, lambda nb_tid,tid=tid: color_of(nb_tid)==color_of(tid))
        else:
            d_me  = delta_attr(self.occ, self.uf8[color_of(tid)],       v, lambda nb_tid,tid=tid: color_of(nb_tid)==color_of(tid))
            d_opp = delta_attr(self.occ, self.uf8[4+symbol_of(tid)],    v, lambda nb_tid,tid=tid: symbol_of(nb_tid)==symbol_of(tid))
        return d_me, d_opp

    # 손패 시너지 & 자유도
    def synergy_and_liberty_bonus(self, v:int, tid:int)->float:
        if self.isFirst:
            sym = symbol_of(tid)
            same_cnt = sum(1 for t in self.myTiles if symbol_of(tile_id(t))==sym)
        else:
            col = color_of(tid)
            same_cnt = sum(1 for t in self.myTiles if color_of(tile_id(t))==col)
        liberties = sum(1 for nb in ADJ[v] if self.occ[nb]==-1)
        return ALPHA_SYNERGY*same_cnt + BETA_LIBERTY*liberties

    # 인접 동일 문양/색상 수
    def count_adj_same_symbol(self, v:int, tid:int)->int:
        s = symbol_of(tid); cnt=0
        for nb in ADJ[v]:
            t = self.occ[nb]
            if t!=-1 and symbol_of(t)==s: cnt += 1
        return cnt
    def count_adj_same_color(self, v:int, tid:int)->int:
        c = color_of(tid); cnt=0
        for nb in ADJ[v]:
            t = self.occ[nb]
            if t!=-1 and color_of(t)==c: cnt += 1
        return cnt

    # 이웃 색상 다양성(선공 보너스용)
    def distinct_neighbor_colors(self, v:int)->int:
        s=set()
        for nb in ADJ[v]:
            t=self.occ[nb]
            if t!=-1: s.add(color_of(t))
        return len(s)

    # 상대가 u에 한 수 둘 때 얻는 "속성별 최대 Δ"
    def opp_local_best_gain(self, u:int)->int:
        if self.isFirst:
            best=0
            for c in range(4):
                g = delta_attr(self.occ, self.uf8[c], u, lambda nb_tid, c=c: color_of(nb_tid)==c)
                if g>best: best=g
            return best
        else:
            best=0
            for s in range(4):
                g = delta_attr(self.occ, self.uf8[4+s], u, lambda nb_tid, s=s: symbol_of(nb_tid)==s)
                if g>best: best=g
            return best

    def local_risk_max_around(self, v:int)->int:
        best=0; checks=0
        for u in ADJ[v]:
            if self.occ[u]==-1:
                best=max(best, self.opp_local_best_gain(u))
                checks+=1
                if checks>=MAX_LOCAL_CHECKS: break
        return best

    # 동적 파라미터(턴/시간/역할)
    def dynamic_params(self, myTime:int):
        # 멍청한 버전은 시간/턴 상관없이 고정된 낮은 값 리턴
        lam = 0.1
        mu  = 0.0
        center_w = 0.0
        center2_w= 0.0
        k_my, k_opp, k_re, k_opp2 = K_MY_DEF, K_OPP_DEF, K_RE_DEF, K_OPP2_DEF
        block_w_mul = 0.0
        thr2 = 999
        return lam, mu, center_w, center2_w, k_my, k_opp, k_re, k_opp2, block_w_mul, thr2

    # 상대 top-K 응수 리스트(현재 상태)
    def opp_top_moves_now(self, opp_ids:List[int], lam:float, k_opp:int)->List[Tuple[int,int,int]]:
        empties=[i for i in range(V) if self.occ[i]==-1]
        if not empties or not opp_ids: return []
        opp_is_first = (not self.isFirst)
        prelim=[]
        for u in empties:
            for otid in opp_ids:
                d_opp = delta_attr(
                    self.occ, self.uf8[4+symbol_of(otid)] if opp_is_first else self.uf8[color_of(otid)],
                    u,
                    (lambda nb_tid, otid=otid: (symbol_of(nb_tid)==symbol_of(otid)) if opp_is_first else (color_of(nb_tid)==color_of(otid)))
                )
                if d_opp<=0: continue
                d_me_from_opp = delta_attr(
                    self.occ, self.uf8[color_of(otid)] if opp_is_first else self.uf8[4+symbol_of(otid)],
                    u,
                    (lambda nb_tid, otid=otid: (color_of(nb_tid)==color_of(otid)) if opp_is_first else (symbol_of(nb_tid)==symbol_of(otid)))
                )
                val = d_opp - lam*d_me_from_opp
                prelim.append((val,u,otid))
        prelim.sort(reverse=True, key=lambda x:x[0])
        return prelim[:min(k_opp,len(prelim))]

    # 내 재응수(최선 1수)
    def my_best_reply_move_now(self, my_ids:List[int], lam:float, k_my:int)->Tuple[int, Optional[Tuple[int,int]]]:
        empties=[i for i in range(V) if self.occ[i]==-1]
        if not empties or not my_ids: return 0, None
        me_first=self.isFirst
        prelim=[]
        for v in empties:
            for tid in my_ids:
                d_me = delta_attr(
                    self.occ, self.uf8[4+symbol_of(tid)] if me_first else self.uf8[color_of(tid)],
                    v,
                    (lambda nb_tid, tid=tid: (symbol_of(nb_tid)==symbol_of(tid)) if me_first else (color_of(nb_tid)==color_of(tid)))
                )
                if d_me>0: prelim.append((d_me,v,tid))
        if not prelim: return 0, None
        prelim.sort(reverse=True, key=lambda x:x[0])
        prelim=prelim[:min(k_my,len(prelim))]

        best=-10**9; best_act=None
        for _,v,tid in prelim:
            d_me = delta_attr(
                self.occ, self.uf8[4+symbol_of(tid)] if me_first else self.uf8[color_of(tid)],
                v,
                (lambda nb_tid, tid=tid: (symbol_of(nb_tid)==symbol_of(tid)) if me_first else (color_of(nb_tid)==color_of(tid)))
            )
            d_opp = delta_attr(
                self.occ, self.uf8[color_of(tid)] if me_first else self.uf8[4+symbol_of(tid)],
                v,
                (lambda nb_tid, tid=tid: (color_of(nb_tid)==color_of(tid)) if me_first else (symbol_of(nb_tid)==symbol_of(tid)))
            )
            val = d_me - lam*d_opp
            if val>best:
                best=val; best_act=(v,tid)
        if best<0: best=0
        return best,best_act

    # 메인 선택
    def calculateMove(self, myTime:int, oppTime:int)->Tuple[Cell, Tile]:
        # 시간 매우 여유: 최대 4.5초, 최소 0.2초
        budget = min(int(myTime*0.95), 4500)
        budget = max(200, budget)
        tg = TimeGuard(budget)

        lam, mu, center_w, center2_w, k_my, k_opp, k_re, k_opp2, block_w_mul, thr2 = self.dynamic_params(myTime)
        cellBias=self.cellBias

        # 후보 생성
        empties=self.frontier_empties()
        if not empties: empties=[i for i in range(V) if self.occ[i]==-1]
        my_ids_all=[tile_id(t) for t in self.myTiles]
        cands=[(v,tid) for v in empties for tid in my_ids_all]
        assert cands, "No legal moves"

        # baseline 상대 위협(블록/Δ패널티용)
        opp_ids_all=[tile_id(t) for t in self.oppTiles]
        baseline_list = self.opp_top_moves_now(opp_ids_all, lam, max(8, k_opp//2))
        baseline_opp = baseline_list[0][0] if baseline_list else 0

        # 1차 빔: 빠른 스코어
        placed = sum(1 for x in self.occ if x!=-1)
        scored=[]
        for v,tid in cands:
            d_me,d_opp = self.eval_delta_both(v, tid)
            extra = self.synergy_and_liberty_bonus(v, tid)

            base = d_me - lam*d_opp + extra + (cellBias[v] if USE_CELL_BIAS else 0.0)
            base += center_w*CENTER[v] + center2_w*CENTER2[v] + DEG_W*DEG[v]

            scored.append((base,v,tid,d_me,d_opp))
        scored.sort(reverse=True, key=lambda x:x[0])
        scored=scored[:min(k_my,len(scored))]

        best_val=-1e100; best_move=None

        for base,v,tid,d_me,d_opp in scored:
            if tg.over(): break
            pre_local = self.local_risk_max_around(v)

            snap=self.snapshot()
            self.apply(v, tid)

            worst_eff_opp = 0
            opp_list = self.opp_top_moves_now(opp_ids_all, lam, k_opp)
            if opp_list:
                for first_val,u,otid in opp_list:
                    if tg.over(): break
                    snap2=self.snapshot()
                    self.apply(u, otid)

                    my_ids_after = multiset_minus_one(my_ids_all, tid)
                    my_reply_val, my_reply_act = self.my_best_reply_move_now(my_ids_after, lam, k_re)

                    eff_opp = max(0.0, first_val - GAMMA_Q*my_reply_val)

                    if eff_opp > worst_eff_opp:
                        worst_eff_opp = eff_opp

                    self.rollback(snap2)

            # 블록/Δ 위협
            block_reward = max(0.0, baseline_opp - worst_eff_opp)
            threat_delta = max(0.0, worst_eff_opp - baseline_opp)

            val = (d_me - lam*d_opp) \
                  - mu*worst_eff_opp \
                  - 0.0*threat_delta \
                  + (BLOCK_W*block_w_mul)*block_reward
            val += (cellBias[v] if USE_CELL_BIAS else 0.0) + 1e-6*(CENTER[v]+DEG[v])

            if val>best_val:
                best_val=val; best_move=(v,tid)

            self.rollback(snap)

        if best_move is None:
            v,tid=scored[0][1], scored[0][2]
        else:
            v,tid=best_move

        cell=ALL_CELLS[v]
        t=Tile(list(Color)[color_of(tid)], list(Symbol)[symbol_of(tid)])
        return cell,t

    # 인터랙션 업데이트
    def updateAction(self, myAction:bool, action:Tuple[Cell, Tile], get:Optional[Tile], _usedTime:Optional[int]):
        cell,tile=action
        v=KEY2IDX[str(cell)]
        tid=tile_id(tile)
        self.apply(v, tid)
        if myAction:
            for i,x in enumerate(self.myTiles):
                if x==tile: self.myTiles.pop(i); break
            if get is not None: self.myTiles.append(get)
        else:
            for i,x in enumerate(self.oppTiles):
                if x==tile: self.oppTiles.pop(i); break
            if get is not None: self.oppTiles.append(get)

# ───────────────────── I/O 루프 ─────────────────────
def main():
    game: Optional[Game]=None
    isFirst: Optional[bool]=None
    lastMove: Optional[Tuple[Cell, Tile]]=None

    while True:
        try:
            line=sys.stdin.readline()
            if not line: break
            line=line.strip()
            if not line: continue
            parts=line.split()
            cmd,args=parts[0],parts[1:]

            if cmd=="READY":
                if len(args)>=1 and args[0] in ("FIRST","SECOND"):
                    isFirst=(args[0]=="FIRST")
                else:
                    isFirst=True
                print("OK", flush=True); continue

            if cmd=="INIT":
                myTiles  = [Tile.from_string(s) for s in args[:5]]
                oppTiles = [Tile.from_string(s) for s in args[5:10]]
                if isFirst is None:
                    isFirst = True
                game=Game(myTiles, oppTiles, isFirst); continue

            if cmd=="TIME":
                myTime,oppTime=int(args[0]),int(args[1])
                assert game is not None
                lastMove=game.calculateMove(myTime, oppTime)
                c,t=lastMove
                print(f"PUT {c} {t}", flush=True); continue

            if cmd=="GET":
                assert game is not None and lastMove is not None
                get = None if args[0]=="X0" else Tile.from_string(args[0])
                game.updateAction(True, lastMove, get, None); continue

            if cmd=="OPP":
                assert game is not None
                cell=Cell.from_string(args[0]); tile=Tile.from_string(args[1])
                get=None if args[2]=="X0" else Tile.from_string(args[2])
                oppTime=int(args[3])
                game.updateAction(False, (cell,tile), get, oppTime); continue

            if cmd=="FINISH":
                break

            continue

        except EOFError:
            break
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__=="__main__":
    main()