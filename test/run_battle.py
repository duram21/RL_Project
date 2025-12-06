import os
import subprocess
import random
import sys
import time

# 1. 입력 파일(input.txt) 생성
def generate_input_file():
    colors = ['R', 'G', 'B', 'Y']
    symbols = ['1', '2', '3', '4']
    base_deck = [c+s for c in colors for s in symbols]
    
    p1_deck = base_deck * 2
    random.shuffle(p1_deck)
    p2_deck = base_deck * 2
    random.shuffle(p2_deck)

    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(p1_deck) + "\n")
        f.write(" ".join(p2_deck) + "\n")

# 2. Config 생성 (경로 문제 방지용)
def generate_config_file():
    python_cmd = sys.executable.replace('\\', '/')
    cwd = os.getcwd().replace('\\', '/')
    
    ppo_script = f"{cwd}/player_ppo.py"
    rnd_script = f"{cwd}/player_random.py"

    # 여기서 LOG는 지정하지 않고, 실행할 때 -l 옵션으로 줍니다 (더 확실함)
    config_content = f"""
INPUT=input.txt
EXEC1="{python_cmd}" "{ppo_script}"
EXEC2="{python_cmd}" "{rnd_script}"
CWD1={cwd}
CWD2={cwd}
"""
    with open("config.ini", "w", encoding="utf-8") as f:
        f.write(config_content.strip())

# 3. 게임 실행 및 파싱
def play_one_game(idx):
    generate_input_file()
    # generate_config_file() # 매번 리셋
    
    # 로그 파일명 지정 (절대 경로)
    log_filename = f"battle_log_{idx:03d}.txt"
    log_path = os.path.abspath(os.path.join("logs", log_filename))
    
    python_cmd = sys.executable 
    tool_script = "testing-tool-connexion.py"

    if not os.path.exists(tool_script):
        return "ERR", "N/A", "Tool Not Found"

    # [수정] -l 옵션으로 로그 경로를 강제로 주입
    cmd = [python_cmd, tool_script, "-c", "config.ini", "-l", log_path]
    
    # 실행 (에러 메시지 캡처를 위해 capture_output=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 🛑 [중요] 로그 파일이 없으면, 툴이 뱉은 에러를 리턴함
    if not os.path.exists(log_path):
        err_msg = result.stderr.strip() if result.stderr else "No Error Msg"
        # 너무 길면 자름
        if len(err_msg) > 50: err_msg = err_msg.split('\n')[-1] 
        return "FAIL", "N/A", f"툴 실행 실패: {err_msg}"

    # 로그 파일 읽기
    with open(log_path, "r", encoding="utf-8") as f:
        log = f.read()
        
        s1, s2 = "0", "0"
        for line in log.splitlines():
            if line.startswith("SCOREFIRST"): s1 = line.split()[1]
            if line.startswith("SCORESECOND"): s2 = line.split()[1]
        
        score_txt = f"{s1} : {s2}"

        if 'RESULT "1-0"' in log: return "WIN", score_txt, log_filename
        elif 'RESULT "0-1"' in log: return "LOSE", score_txt, log_filename
        elif 'RESULT "1/2-1/2"' in log: return "DRAW", score_txt, log_filename
        elif "ABORT" in log: 
            try: reason = log.split("ABORT")[-1].strip().splitlines()[0]
            except: reason = "Unknown Abort"
            return "ABORT", score_txt, f"{reason} ({log_filename})"
        else: return "UNKNOWN", score_txt, log_filename

# 4. 메인 루프
def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    while True:
        try:
            n_input = input("🎮 몇 판을 진행하시겠습니까? (숫자 입력): ")
            num_games = int(n_input)
            break
        except ValueError:
            print("숫자를 입력해주세요.")

    print(f"\n🚀 {num_games}판 대결 시작! (PPO vs Random)\n")
    print("-" * 80)
    print(f"{'No.':<6} | {'Result':<8} | {'Score':<15} | {'Detail / Error Msg'}")
    print("-" * 80)

    wins = 0; losses = 0; draws = 0; errors = 0

    start_time = time.time()

    for i in range(1, num_games + 1):
        res_type, score_str, detail = play_one_game(i)
        
        icon = "❓"
        if res_type == "WIN": wins += 1; icon = "🏆 WIN"
        elif res_type == "LOSE": losses += 1; icon = "💀 LOSE"
        elif res_type == "DRAW": draws += 1; icon = "🤝 DRAW"
        else:
            errors += 1
            icon = f"💥 {res_type}"

        print(f"#{i:<5} | {icon:<8} | {score_str:<15} | {detail}")

    end_time = time.time()
    duration = end_time - start_time

    print("-" * 80)
    print(f"⏱️  소요 시간: {duration:.2f}초")
    print(f"📊 최종 전적: {wins}승 {draws}무 {losses}패 (에러 {errors})")
    if num_games > 0:
        print(f"📈 승률: {(wins/num_games*100):.1f}%")
    print(f"📂 로그 저장소: {os.path.abspath('logs')}")
    print("=" * 80)

if __name__ == "__main__":
    main()