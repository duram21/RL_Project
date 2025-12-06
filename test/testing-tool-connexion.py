#!/usr/bin/env python3

import argparse
from collections import Counter
from enum import Enum
import io
import json
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
import threading
from typing import List, Optional, TextIO, Tuple


class AbortError(Exception):
    """
    Exception raised when the game is aborted due to invalid input.
    """


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


@dataclass
class Cell:
    """
    Dataclass that represents a cell on the board.
    """

    col: Column
    row: Row
    sign: Sign

    def __str__(self):
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
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
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)

        return (
            self.sign == Sign.MINUS
            and other.sign == Sign.PLUS
            and (dr, dc) in [(0, 0), (0, -1), (-1, 0)]
        ) or (
            self.sign == Sign.PLUS
            and other.sign == Sign.MINUS
            and (dr, dc) in [(0, 0), (0, 1), (1, 0)]
        )

    @classmethod
    def fromString(cls, s: str) -> "Cell":
        if len(s) != 3:
            raise AbortError("CELL PARSE FAILED")
        try:
            cell = cls(Column(s[0]), Row(s[1]), Sign(s[2]))
        except ValueError:
            raise AbortError("CELL PARSE FAILED")
        if not cell.isValid():
            raise AbortError("CELL PARSE FAILED")
        return cell


# Add frozen to use hash in Counter
@dataclass(frozen=True)
class Tile:
    """
    Dataclass that represents a tile.
    """

    color: Color
    symbol: Symbol

    def __str__(self):
        return f"{self.color.value}{self.symbol.value}"

    def isSame(self, other: "Tile", first: bool) -> bool:
        return self.symbol == other.symbol if first else self.color == other.color

    @classmethod
    def fromString(cls, s: str) -> Optional["Tile"]:
        if len(s) != 2:
            raise AbortError("TILE PARSE FAILED")
        if s == "X0":
            return None
        try:
            return cls(Color(s[0]), Symbol(s[1]))
        except ValueError:
            raise AbortError("TILE PARSE FAILED")


@dataclass
class Settings:
    """
    Settings for connexion game
    - inputData: tuple of (list of tiles for first player, list of tiles for second player)
    - exec1: command to execute first player
    - exec2: command to execute second player
    """

    inputData: Tuple[List[Tile], List[Tile]]
    exec1: str
    exec2: str
    cwd1: Optional[str]
    cwd2: Optional[str]


def parsePut(s: str) -> Tuple[Cell, Tile]:
    try:
        putStr, cellStr, tileStr = s.split()
        if putStr != "PUT":
            raise AbortError("PUT PARSE FAILED")
        cell, tile = Cell.fromString(cellStr), Tile.fromString(tileStr)
        if tile is None:
            raise AbortError("TILE PARSE FAILED")
        return cell, tile
    except ValueError:
        raise AbortError("PUT PARSE FAILED")


class Game:
    """
    Game from `sample-code.py`, with some modifications on raising exceptions, and first/second player distinction.
    """

    def __init__(self, firstTiles: List[Tile], secondTiles: List[Tile]):
        self.firstTiles = firstTiles
        self.secondTiles = secondTiles
        self.board: List[Tuple[Cell, Tile]] = []

    def updateAction(self, first: bool, action: Tuple[Cell, Tile], get: Optional[Tile]):
        ac, at = action
        if any(c == ac for c, _ in self.board):
            raise AbortError("NOT EMPTY")

        self.board.append(action)
        if first:
            try:
                self.firstTiles.remove(at)
            except ValueError:
                raise AbortError("NO TILE")
            if get is not None:
                self.firstTiles.append(get)
        else:
            try:
                self.secondTiles.remove(at)
            except ValueError:
                raise AbortError("NO TILE")
            if get is not None:
                self.secondTiles.append(get)

    @classmethod
    def calculateScore(cls, board: List[Tuple[Cell, Tile]], first: bool) -> int:
        if not board:
            return 0

        adj = [[False] * len(board) for _ in range(len(board))]
        for i, (ci, ti) in enumerate(board):
            for j, (cj, tj) in enumerate(board):
                if i == j or (ci.isAdjacent(cj) and ti.isSame(tj, first)):
                    adj[i][j] = True

        for k in range(len(board)):
            for i in range(len(board)):
                for j in range(len(board)):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True

        return sum(sum(row) for row in adj)


def runGame(settings: Settings, res: TextIO):
    """
    Run the game from settings, and returns the result as a string.
    """

    def p(x):
        return ["FIRST", "SECOND"][x]

    e1 = json.dumps(f"COMMAND: {settings.exec1}", ensure_ascii=False)
    e2 = json.dumps(f"COMMAND: {settings.exec2}", ensure_ascii=False)
    res.write(f"[{p(0)} {e1}]\n[{p(1)} {e2}]\n")

    f = io.StringIO()
    users = [Player(0, settings.exec1, settings.cwd1, f),
             Player(1, settings.exec2, settings.cwd2, f)]
    result = "*"

    try:
        # Ready phase
        users[0].print("READY FIRST")
        users[1].print("READY SECOND")
        lines = Player.readAll(users, 3.0)
        for i, line in enumerate(lines):
            if line is None:
                f.write(f"ABORT {p(i)} TLE\n")
                result = f"{str(i)}-{str(1-i)}"
                return
            if line.strip() != "OK":
                f.write(f"ABORT {p(i)} INVALID READY MESSAGE\n")
                result = f"{str(i)}-{str(1-i)}"
                return

        # Init
        firstTiles = settings.inputData[0][:5]
        secondTiles = settings.inputData[1][:5]
        firstTileStr = " ".join(map(str, firstTiles))
        secondTileStr = " ".join(map(str, secondTiles))
        users[0].print(f"INIT {firstTileStr} {secondTileStr}")
        users[1].print(f"INIT {secondTileStr} {firstTileStr}")
        f.write(f"INIT {firstTileStr} {secondTileStr}\n")

        # Proceed turn
        time = [10_000, 10_000]
        game = Game(firstTiles, secondTiles)
        for round in range(64):
            # notify time to user
            u = round % 2
            users[u].print(f"TIME {time[u]} {time[1-u]}")
            resp = users[u].readline(time[u] / 1000)
            if resp is None:
                f.write(f"ABORT {p(u)} TLE\n")
                result = f"{str(u)}-{str(1-u)}"
                return

            # deduct time
            tm, putStr = resp
            usedTime = min(time[u], int(tm * 1000))
            time[u] -= usedTime

            # parse, draw, and notify
            drawTile = settings.inputData[u][round //
                                             2 + 5] if round < 54 else None
            drawTileStr = "X0" if drawTile is None else str(drawTile)
            try:
                cell, tile = parsePut(putStr)
                game.updateAction(u == 0, (cell, tile), drawTile)
            except AbortError as e:
                f.write(f"ABORT {p(u)} {e.args[0]}\n")
                result = f"{str(u)}-{str(1-u)}"
                return
            users[u].print(f"GET {drawTileStr}")
            users[1 - u].print(f"OPP {cell} {tile} {drawTileStr} {usedTime}")
            f.write(f"{p(u)} {cell} {tile} {drawTileStr} {usedTime}\n")

        # Game is finished
        f.write("FINISH\n")
        score0, score1 = game.calculateScore(game.board, True), game.calculateScore(
            game.board, False
        )
        f.write(f"SCOREFIRST {score0}\n")
        f.write(f"SCORESECOND {score1}\n")
        if score0 > score1:
            result = f"1-0"
        elif score0 < score1:
            result = f"0-1"
        else:
            result = f"1/2-1/2"

    finally:
        for u in users:
            u.print("FINISH")
            u.join()
        res.write(f'[RESULT "{result}"]\n')
        res.write(f.getvalue())


class Player:
    """
    Process for a player, that supports rw to stdin/stdout/stderr and terminating the process".
    """

    def __init__(self, no: int, exec: str, cwd: Optional[str], logStream: TextIO):
        self.name = ["FIRST", "SECOND"][no]
        self.exec = exec
        try:
            self.process = subprocess.Popen(
                self.exec,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                cwd=cwd,
            )
        except Exception as e:
            print(f"Error: failed to start process {self.exec}: {e}")
            sys.exit(1)

        self.reads = queue.Queue()
        self.writes = queue.Queue()
        self.logStream = logStream

        self.stdin_thread = threading.Thread(target=self.__handle_stdin)
        self.stdout_thread = threading.Thread(target=self.__handle_stdout)
        self.stderr_thread = threading.Thread(target=self.__handle_stderr)
        self.stdin_thread.start()
        self.stdout_thread.start()
        self.stderr_thread.start()

    def __handle_stdin(self):
        """
        Write to the player's stdin.
        """
        stdin = self.process.stdin
        assert stdin is not None
        try:
            while True:
                res = self.writes.get()
                if res is None:
                    break
                stdin.write(f"{res}\n")
                stdin.flush()
        finally:
            stdin.close()

    def __handle_stdout(self):
        """
        Read from the player's stdout, and put the result to the queue.
        """
        stdout = self.process.stdout
        assert stdout is not None
        try:
            while True:
                r = stdout.readline()
                if not r:
                    break
                self.reads.put(r)
        except:
            pass
        finally:
            stdout.close()

    def __handle_stderr(self):
        """
        Read from the player's stderr, and redirect to the log file.
        """
        stderr = self.process.stderr
        assert stderr is not None
        try:
            while True:
                r = stderr.readline()
                if not r:
                    break
                self.logStream.write(f"# Debug {self.name}: {r.rstrip()}\n")
        except:
            pass
        finally:
            stderr.close()

    def print(self, message: str):
        """
        Print a message to the player's stdin. Newline is added automatically.
        """
        self.writes.put(message)

    def readline(self, timeout: float) -> Optional[Tuple[float, str]]:
        """
        Read a line from the player's stdout.
        Return None if timeout.
        """
        try:
            start = time.perf_counter()
            result = self.reads.get(timeout=timeout)
            return min(timeout, time.perf_counter() - start), result
        except queue.Empty:
            return None

    @classmethod
    def readAll(cls, selfs: List["Player"], timeout: float) -> List[Optional[str]]:
        """
        Read all lines from the players' stdout.
        Return None if timeout.
        """

        def __readline_thread(
            p: "Player", timeout: float, idx: int, arr: List[Optional[str]]
        ):
            ret = p.readline(timeout)
            if ret is None:
                arr[idx] = None
            else:
                arr[idx] = ret[1]

        readline_threads = []
        returns: List[Optional[str]] = [None] * len(selfs)
        for i, p in enumerate(selfs):
            readline_threads.append(
                threading.Thread(
                    target=__readline_thread, args=(p, timeout, i, returns)
                )
            )

        for thread in readline_threads:
            thread.start()

        for thread in readline_threads:
            thread.join()

        return returns

    def join(self, timeout: Optional[float] = 1.0):
        """
        Join the player's process.
        If timeout, terminate the process (SIGTERM in POSIX) and wait for it to exit.
        If the process is not terminated within timeout, kill it.
        """
        self.writes.put(None)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def readInput(f: TextIO) -> Tuple[List[Tile], List[Tile]]:
    """
    Read input data from file.
    """

    inputData = []
    for _ in range(2):
        try:
            tiles = f.readline().split()
            if len(tiles) != 32:
                raise ValueError("Invalid input: user tiles must be 32")
            try:
                parsedTiles = [Tile.fromString(tile) for tile in tiles]
            except AbortError:
                raise ValueError("Invalid tile is provided")
            if None in parsedTiles:
                raise ValueError("Invalid tile is provided")
            inputData.append(parsedTiles)

            if any(x != 2 for x in Counter(parsedTiles).values()):
                raise ValueError(
                    "Each (color, symbol) pair must be provided exactly twice"
                )

        except Exception as e:
            print(f"Invalid input file: {e}", file=sys.stderr)
            sys.exit(1)

    return tuple(inputData)


def readSettings() -> Tuple[TextIO, Settings]:
    """
    Read configs from command line arguments or config file.
    Return (logFile, settings) tuple.
    """

    parser = argparse.ArgumentParser(
        prog="testing-tool-connexion",
        description="Testing tool for connexion",
        epilog="For detailed information, please see README.md.",
    )

    parser.add_argument("-c", "--config", type=str,
                        help="predefined config file")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file",
    )
    parser.add_argument("-l", "--log", type=str, help="Log file")

    parser.add_argument(
        "-s",
        "--stdio",
        nargs="?",
        const=True,
        type=lambda x: True if x is None else x.lower() == "true",
        default=False,
        help="Use stdandard input/output for input and log file",
    )

    parser.add_argument("-a", "--exec1", type=str, help="First player command")
    parser.add_argument("-b", "--exec2", type=str,
                        help="Second player command")
    parser.add_argument("--cwd1", type=str,
                        help="First player working directory")
    parser.add_argument("--cwd2", type=str,
                        help="Second player working directory")

    args = parser.parse_args()

    inputFile = args.input
    logFile = args.log
    exec1 = args.exec1
    exec2 = args.exec2
    stdio = args.stdio
    cwd1 = args.cwd1
    cwd2 = args.cwd2

    # Read key=value settings
    if args.config:
        try:
            f = open(args.config, "r", encoding="utf-8")
        except FileNotFoundError:
            parser.print_help()
            print(f"\nError: Config file {args.config} not found.")
            sys.exit(1)

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                if key == "INPUT":
                    if inputFile is None:
                        inputFile = value
                elif key == "LOG":
                    if logFile is None:
                        logFile = value
                elif key == "EXEC1":
                    if exec1 is None:
                        exec1 = value
                elif key == "EXEC2":
                    if exec2 is None:
                        exec2 = value
                elif key == "CWD1":
                    if cwd1 is None:
                        cwd1 = value
                elif key == "CWD2":
                    if cwd2 is None:
                        cwd2 = value
                else:
                    parser.print_help()
                    print(f"\nUnknown line: {line}", file=sys.stderr)
                    sys.exit(1)
            else:
                parser.print_help()
                print(f"\nUnknown line: {line}", file=sys.stderr)
                sys.exit(1)
        f.close()

    # Specify file stream
    if not inputFile:
        if not stdio:
            parser.print_help()
            print("\nError: No input file provided.", file=sys.stderr)
            sys.exit(1)
        else:
            f = sys.stdin
            def close(): return None
    else:
        try:
            f = open(inputFile, "r", encoding="utf-8")
            def close(): return f.close()

        except FileNotFoundError:
            parser.print_help()
            print(
                f"\nError: Input file {inputFile} not found.", file=sys.stderr)
            sys.exit(1)

    # Read input
    inputData = readInput(f)
    close()

    if not stdio and not logFile:
        parser.print_help()
        print("\nError: No log output file provided.", file=sys.stderr)
        sys.exit(1)

    if not exec1:
        parser.print_help()
        print("\nError: First player command not specified.", file=sys.stderr)
        sys.exit(1)
    if not exec2:
        parser.print_help()
        print("\nError: Second player command not specified.", file=sys.stderr)
        sys.exit(1)

    if logFile is None:
        logStream = sys.stdout
    else:
        try:
            logStream = open(logFile, "w", encoding="utf-8")
        except FileNotFoundError:
            parser.print_help()
            print(f"\nError: Log file {logFile} not found.", file=sys.stderr)
            sys.exit(1)

    return (logStream, Settings(inputData, exec1, exec2, cwd1, cwd2))


def main():
    logStream, settings = readSettings()
    runGame(settings, logStream)
    logStream.close()


if __name__ == "__main__":
    main()
