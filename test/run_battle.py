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


# 3. 게임 실행 및 파싱
def play_one_game(idx):
    generate_input_file()

    
    # 로그 파일명 지정 (절대 경로)
    log_filename = f"battle_log_{idx:03d}.txt"
    log_path = os.path.abspath(os.path.join("logs", log_filename))
    
    python_cmd = sys.executable 
    tool_script = "testing-tool-connexion.py"

    if not os.path.exists(tool_script):
        return "ERR", "N/A", "Tool Not Found"

    # [수정] -l 옵션으로 로그 경로를 강제로 주입
    cmd = [python_cmd, tool_script, "-c", "config.ini", "-l", log_path]
    
    # 실행
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 🛑 [중요] 로그 파일이 없으면, 툴이 뱉은 에러를 리턴함
    if not os.path.exists(log_path):
        err_msg = result.stderr.strip() if result.stderr else "No Error Msg"
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

    print(f"\n🚀 {num_games}판 대결 시작! (PPO vs )\n")
    
    # 저장할 결과 문자열들을 모을 리스트
    summary_lines = []
    
    # 헤더 생성
    header_line_1 = "-" * 80
    header_line_2 = f"{'No.':<6} | {'Result':<8} | {'Score':<15} | {'Detail / Error Msg'}"
    
    # 화면 출력
    print(header_line_1)
    print(header_line_2)
    print(header_line_1)
    
    # 저장용 리스트에 추가
    summary_lines.append(header_line_1)
    summary_lines.append(header_line_2)
    summary_lines.append(header_line_1)

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

        # 결과 문자열 생성
        row_str = f"#{i:<5} | {icon:<8} | {score_str:<15} | {detail}"
        
        # 화면 출력 및 저장
        print(row_str)
        summary_lines.append(row_str)

    end_time = time.time()
    duration = end_time - start_time

    # 최종 통계 생성
    footer_lines = [
        "-" * 80,
        f"⏱️  소요 시간: {duration:.2f}초",
        f"📊 최종 전적: {wins}승 {draws}무 {losses}패 (에러 {errors})"
    ]
    
    if num_games > 0:
        footer_lines.append(f"📈 승률: {(wins/num_games*100):.1f}%")
    
    # footer_lines.append(f"📂 로그 저장소: {os.path.abspath('logs')}")
    footer_lines.append("=" * 80)

    # 통계 화면 출력 및 저장
    for line in footer_lines:
        print(line)
        summary_lines.append(line)

    # 🔥 파일로 저장
    summary_file = "battle_summary.txt"
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\n✅ 결과 요약본이 '{summary_file}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 결과 파일 저장 실패: {e}")

if __name__ == "__main__":
    main()