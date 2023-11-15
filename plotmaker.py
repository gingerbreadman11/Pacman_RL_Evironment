import re
import matplotlib.pyplot as plt
import os

def extract_q_values(log_file_name):
    q_values = []
    with open(log_file_name, 'r') as file:
        for i, line in enumerate(file):
            if i<50000:
                if i % 100 == 0:  # Only read every nth line
                    match = re.search(r'Q:\s*([-\d.]+)', line)
                    if match:
                        q_value = float(match.group(1))
                        q_values.append(q_value)
    return q_values

def extract_r_values(log_file_name):
    r_values = []
    with open(log_file_name, 'r') as file:
        for i, line in enumerate(file):
            if i<50000:
                if i % 100 == 0:  # Only read every nth line
                    match = re.search(r'r:\s*([-\d.]+)', line)
                    if match:
                        r_value = float(match.group(1))
                        r_values.append(r_value)
    return r_values

def plot_q_values(q_values, title):
    plt.plot(range(1, len(q_values) + 1), q_values, marker='o')
    plt.xlabel('Episodes X 100')
    plt.ylabel('Q Value')
    plt.title(title)
    # Save plot
    plt.savefig(title + '_Q.pdf')
    plt.show()

def plot_r_values(r_values, title):
    plt.plot(range(1, len(r_values) + 1), r_values, marker='o')
    plt.xlabel('Episodes X 100')
    plt.ylabel('reward Value')
    plt.title(title)
    # Save plot
    plt.savefig(title + '_r.pdf')
    plt.show()

# Update this line with the relative path to your log file within the logs folder
log_file_name = '/Users/alexanderbensland/Desktop/Code/ESP3201/AI_2.0/upgraded_Pacman_RL_Evironment/logs/DQN4_mediumGrid.log'
q_values = extract_q_values(log_file_name)
r_values = extract_r_values(log_file_name)
# Extracting the file name from the path for the title
#file_title = os.path.basename(log_file_name)
#plot_q_values(q_values, file_title)

# Extracting the file name from the path and removing the '.log' extension for the title
file_title = os.path.basename(log_file_name).replace('.log', '')
plot_q_values(q_values, file_title)
plot_r_values(r_values, file_title)