import matplotlib.pyplot as plt

# Path to your log file
log_file_path = '/Users/alexanderbensland/Desktop/Code/ESP3201/AI_2.0/upgraded_Pacman_RL_Evironment/logs/DQN4_mediumGrid.log'

# Initialize variables
recent_wins = []
win_rates = []

# Read the log file
with open(log_file_path, 'r') as file:
    for line in file:
        # Check if the game was won
        won = 'won: True' in line

        # Update the recent wins list
        recent_wins.append(won)
        if len(recent_wins) > 100:
            recent_wins.pop(0)

        # Calculate the average win rate for the last 10 games
        avg_win_rate = sum(recent_wins) / len(recent_wins)
        win_rates.append(avg_win_rate)

# Plotting
plt.plot(range(1, len(win_rates) + 1), win_rates)
plt.xlabel('episodes')
plt.ylabel('Moving Average Win Rate (Last 100 Entries)')
plt.title('4LayerDQN Win Rate')
plt.savefig('4LayerDQN Win Rate.pdf')
plt.show()


