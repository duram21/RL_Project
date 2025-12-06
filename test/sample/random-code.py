# random_agent.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
import random


class Column(Enum):
    """게임 보드의 열 번호"""

    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"


class Row(Enum):
    """게임 보드의 행 번호"""

    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"
    _5 = "5"
    _6 = "6"


class Sign(Enum):
    """게임 보드의 부호"""

    MINUS = "-"
    PLUS = "+"


class Color(Enum):
    """게임 타일의 색"""

    R = "R"
    G = "G"
    B = "B"
    Y = "Y"


class Symbol(Enum):
    """게임 타일의 문양"""

    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"


@dataclass(frozen=True)
class Cell:
    """게임 보드의 칸"""

    col: Column  # 게임 보드의 열
    row: Row     # 게임 보드의 행
    sign: Sign   # 게임 보드의 부호

    def __str__(self):
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
        """
        해당 칸이 금지된 칸인 a1-, a4-, c3+, c6+, d1-, d4-, f3+, f6+ 중 하나가 아니라면 True를 반환
        """
        return self not in [
            Cell(Column.A, Row._1, Sign.MINUS),
            Cell(Column.A, Row._4, Sign.MINUS),
            Cell(Column.C, Row._3, Sign.PLUS),
            Cell(Column.C, Row._6, Sign.PLUS),
            Cell(Column.D, Row._1, Sign.MINUS),
            Cell(Column.D, Row._4, Sign.MINUS),
            Cell(Column.F, Row._3, Sign.PLUS),
            Cell(Column.F, Row._6, Sign.PLUS),
        ]

    def isAdjacent(self, other: "Cell") -> bool:
        """
        주어진 두 칸이 인접한 칸인지 여부를 반환
        """

        # 두 칸의 행, 열 번호 차이를 계산
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)

        return (
            self.sign == Sign.MINUS
            and other.sign == Sign.PLUS
            and (dr == dc == 0 or dr == 0 and dc == -1 or dr == -1 and dc == 0)
        ) or (
            self.sign == Sign.PLUS
            and other.sign == Sign.MINUS
            and (dr == dc == 0 or dr == 0 and dc == 1 or dr == 1 and dc == 0)
        )

    @classmethod
    def getAllCells(cls) -> List["Cell"]:
        """
        배치 가능한 모든 칸을 반환
        """
        ret = []
        for col in Column:
            for row in Row:
                for sign in Sign:
                    cell = cls(col, row, sign)
                    if cell.isValid():
                        ret.append(cell)
        return ret

    @classmethod
    def fromString(cls, s: str) -> "Cell":
        assert len(s) == 3
        cell = cls(Column(s[0]), Row(s[1]), Sign(s[2]))
        assert cell.isValid()
        return cell


@dataclass
class Tile:
    """
    게임 타일
    """

    color: Color  # 타일의 색
    symbol: Symbol  # 타일의 문양

    def __str__(self):
        return f"{self.color.value}{self.symbol.value}"

    def isSame(self, other: "Tile", isFirst: bool) -> bool:
        """
        점수 계산시 두 타일을 같은 타일로 볼 것인가를 반환

        인자 목록
        - other (Tile): 비교할 타일
        - isFirst (bool): 선후공 (True: 선공 / 문양을 사용 , False: 후공 / 색을 사용)
        """
        return self.symbol == other.symbol if isFirst else self.color == other.color

    @classmethod
    def fromString(cls, s: str) -> "Tile":
        assert len(s) == 2
        return cls(Color(s[0]), Symbol(s[1]))


class Game:
    """
    게임 상태를 관리하는 클래스

    변수 목록
    - myTiles (List[Tile]): 내 타일 목록
    - oppTiles (List[Tile]): 상대 타일 목록
    - isFirst (bool): 선후공 (True: 선공, False: 후공)
    - board (List[Tuple[Cell, Tile]]): 현재 보드에 배치된 칸과 타일 목록
    """

    def __init__(self, myTiles: List[Tile], oppTiles: List[Tile], isFirst: bool):
        """
        인자 목록
        - myTiles (List[Tile]): 내 타일 목록
        - oppTiles (List[Tile]): 상대 타일 목록
        - isFirst (bool): 선후공 (True: 선공, False: 후공)
        """

        self.myTiles = myTiles
        self.oppTiles = oppTiles
        self.isFirst = isFirst
        self.board: List[Tuple[Cell, Tile]] = []

    # ================================ [랜덤 에이전트 구현] ================================
    def calculateMove(self, _myTime: int, _oppTime: int) -> Tuple[Cell, Tile]:
        """
        현재 상태를 기반으로 "랜덤하게" 놓을 칸과 타일을 계산함
        """

        # 이미 사용된 칸들
        occupied = {c for c, _ in self.board}  # Cell이 frozen dataclass라 set에 들어갈 수 있음
        # 남은 칸만 후보
        available_cells = [c for c in Cell.getAllCells() if c not in occupied]

        assert self.myTiles, "손패가 비었는데 내 차례가 온 경우는 없어야 함"
        assert available_cells, "둘 수 있는 칸이 없는 경우는 없어야 함"

        tile = random.choice(self.myTiles)
        cell = random.choice(available_cells)
        return cell, tile

    # ============================== [샘플 코드 그대로 유지] ==============================

    def updateAction(
        self,
        myAction: bool,
        action: Tuple[Cell, Tile],
        get: Optional[Tile],
        _usedTime: Optional[int],
    ):
        """
        자신 혹은 상대의 행동을 기반으로 상태를 업데이트 함

        인자 목록
        - myAction (bool): 자신의 행동 여부 (True: 자신, False: 상대)
        - action (Tuple[Cell, Tile]): 자신 혹은 상대가 배치한 칸과 타일
        - get (Optional[Tile]): 자신 혹은 상대가 뽑은 타일. 없으면 None.
        - _usedTime (Optional[int]): 상대가 사용한 시간. 자신의 행동인 경우 None.
        """

        # 전체 보드에 칸과 타일 추가
        self.board.append(action)

        _, tile = action
        if myAction:
            # 내 타일에서 사용한 타일 제거
            self.myTiles.remove(tile)

            # 뽑아온 타일이 있으면 타일 목록에 타일 추가
            if get is not None:
                self.myTiles.append(get)
        else:
            # 상대 타일 목록에서 사용한 타일 제거
            self.oppTiles.remove(tile)

            # 뽑아온 타일이 있으면 타일 목록에 타일 추가
            if get is not None:
                self.oppTiles.append(get)

    @classmethod
    def calculateScore(cls, board: List[Tuple[Cell, Tile]], isFirst: bool) -> int:
        """
        보드와 선후공를 기반으로 점수를 계산함

        인자 목록
        - board (List[Tuple[Cell, Tile]]): 보드
        - isFirst (bool): 선후공 (True: 선공 / 문양을 사용, False: 후공 / 색을 사용)
        """

        if not board:
            return 0

        # 선후공을 고려하여 타일들의 인접행렬을 계산함
        adj = [[False] * len(board) for _ in range(len(board))]
        for i, (ci, ti) in enumerate(board):
            for j, (cj, tj) in enumerate(board):
                if i == j or (ci.isAdjacent(cj) and ti.isSame(tj, isFirst)):
                    adj[i][j] = True

        # Warshall algorithm을 사용하여 연결 성분을 계산
        for k in range(len(board)):
            for i in range(len(board)):
                for j in range(len(board)):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True

        # 각 타일의 점수는 같은 연결 성분에 속한 타일의 수, 총점은 모든 타일의 점수 총합
        return sum(sum(row) for row in adj)


def main():
    game = None
    isFirst = None
    lastMove = None

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                # 게임 시작
                isFirst = args[0] == "FIRST"
                print("OK")
                sys.stdout.flush()
                continue

            if command == "INIT":
                # 준비 단계 시작
                myTiles = [t for t in map(Tile.fromString, args[:5]) if t is not None]
                oppTiles = [t for t in map(Tile.fromString, args[5:]) if t is not None]
                assert isFirst is not None
                game = Game(myTiles, oppTiles, isFirst)
                continue

            if command == "TIME":
                # 배치 단계 시작
                myTime, oppTime = int(args[0]), int(args[1])
                assert game is not None
                lastMove = game.calculateMove(myTime, oppTime)
                cell, tile = lastMove
                print(f"PUT {cell} {tile}", flush=True)
                continue

            if command == "GET":
                # 배치 단계가 끝날때 뽑아온 타일을 이용해서 상태를 업데이트 함
                get = None if args[0] == "X0" else Tile.fromString(args[0])
                assert game is not None and lastMove is not None
                game.updateAction(True, lastMove, get, None)
                continue

            if command == "OPP":
                # 상대의 행동을 기반으로 상태를 업데이트 함
                cell, tile, get, oppTime = (
                    Cell.fromString(args[0]),
                    Tile.fromString(args[1]),
                    None if args[2] == "X0" else Tile.fromString(args[2]),
                    int(args[3]),
                )
                assert game is not None and tile is not None
                game.updateAction(False, (cell, tile), get, oppTime)
                continue

            if command == "FINISH":
                # 게임 종료
                break

            # 알 수 없는 명령어 처리
            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()
