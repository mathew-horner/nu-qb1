#!/usr/bin/env python3
import warnings

warnings.filterwarnings("ignore")

import re
import requests
import sys

TEAM_COUNT_FBS_2024 = 134
TEAM_COUNT_FBS_2025 = 136

if len(sys.argv) != 2:
    print("Usage: python3 defense_rating.py <TEAM>")
    exit(1)

school = sys.argv[1]

page_2024 = requests.get(f"https://www.sports-reference.com/cfb/schools/{school}/2024.html").text
page_2025 = requests.get(f"https://www.sports-reference.com/cfb/schools/{school}/2025.html").text

def extract_game_count(page_content: str) -> int:
    match = re.search("<strong>Record:</strong> (\d*)-(\d*)", page_content)
    wins = int(match.group(1))
    losses = int(match.group(2))
    return wins + losses

def extract_opp_ppg_rank(page_content: str) -> int:
    match = re.search("<strong>Opp Pts/G:</strong> \d*\.\d* \((\d*).*\)", page_content)
    return int(match.group(1))

game_count_2024 = extract_game_count(page_2024)
game_count_2025 = extract_game_count(page_2025)

opp_ppg_2024_rank = extract_opp_ppg_rank(page_2024)
opp_ppg_2025_rank = extract_opp_ppg_rank(page_2025)

rating_2024 = 1 - (opp_ppg_2024_rank / TEAM_COUNT_FBS_2024)
rating_2025 = 1 - (opp_ppg_2025_rank / TEAM_COUNT_FBS_2025)

rating = ((rating_2024 * game_count_2024) + (rating_2025 * game_count_2025)) / (game_count_2024 + game_count_2025)
print(rating)

