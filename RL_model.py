#!/usr/bin/env python3
"""
compare_controllers.py

Compare Fixed-time, SOTL, MaxPressure, and RL (Double DQN) controllers in SUMO.
Outputs:
 - compare_commute.png  (bar chart comparing average commute time)
 - compare_wait.png     (bar chart comparing average wait per vehicle)

Configure SUMO paths and intersection lanes below.
"""

import os
import sys
import time
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- USER CONFIG --------------------
SUMOCFG_FILE = r"path to sumo configuration file"  
TRAFFIC_LIGHT_ID = "INTERSECTION ID ON SUMO"                           
INCOMING_LANES = ["Ids of lanes connected to intersection"] 

# runtime options
GUI = False
EPISODES = 700          # how many evaluation episodes per controller (increase for better stats)
MIN_GREEN_TIME = 20      # seconds per decision window (avoid very short switching)
MAX_SIM_SECONDS = 3600   # safety cap per episode (sim seconds)
LOG_DIR = "logs_compare1"
SEED = 42

# RL hyperparameters
STATE_SIZE = len(INCOMING_LANES) * 2 + 2  # queues + vehicle counts + curr_phase + phase_timer
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 500
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_EPISODES = 300
WARMUP_STEPS = 2000  # steps before learning begins (environment steps)
REPLAY_START_SIZE = 1000

# SOTL hyperparams
SOTL_MU = 50
SOTL_NU = 3
SOTL_PSI = 80
SOTL_OMEGA = 25

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(LOG_DIR, exist_ok=True)

# -------------------- IMPORT TRACI (SUMO) --------------------
def ensure_sumo_path():
    if 'SUMO_HOME' not in os.environ:
        raise RuntimeError("Please set SUMO_HOME environment variable to your SUMO installation path.")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

ensure_sumo_path()
try:
    import traci
except Exception as e:
    raise RuntimeError("traci import failed. Ensure SUMO and SUMO_HOME are configured.") from e

# -------------------- UTILITIES --------------------
def start_sumo(extra_args=None):
    """
    Start SUMO for a single run. Optionally pass a list of extra commandline args like ["--seed","123"].
    """
    ensure_sumo_path()
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui' if GUI else 'sumo')
    cmd = [sumo_binary, "-c", SUMOCFG_FILE]
    if extra_args:
        cmd += extra_args
    traci.start(cmd)

def safe_traci_close():
    try:
        traci.close()
    except Exception:
        pass

def compute_avg_wait_per_vehicle(lanes=INCOMING_LANES):
    total_wait = 0.0
    total_veh = 0
    for l in lanes:
        try:
            total_wait += traci.lane.getWaitingTime(l)
            total_veh += traci.lane.getLastStepVehicleNumber(l)
        except Exception:
            pass
    return float(total_wait / max(1, total_veh))

# -------------------- CONTROLLERS --------------------
def run_fixed_time(ep_seed=None):
    """
    Run SUMO using network's default (fixed) traffic light programs.
    Returns avg_commute_time, avg_wait
    """
    try:
        seed_arg = ["--seed", str(ep_seed)] if ep_seed is not None else None
        start_sumo(seed_arg)
        depart_times = {}
        arrive_times = {}
        wait_samples = []

        # run until no vehicles or safety cap
        while traci.simulation.getTime() < MAX_SIM_SECONDS and traci.simulation.getMinExpectedNumber() > 0:
            t = traci.simulation.getTime()
            for vid in traci.simulation.getDepartedIDList():
                depart_times.setdefault(vid, t)
            for vid in traci.simulation.getArrivedIDList():
                arrive_times.setdefault(vid, t)
            wait_samples.append(compute_avg_wait_per_vehicle())
            traci.simulationStep()

        commute_list = [arrive_times[v] - depart_times[v] for v in depart_times if v in arrive_times]
        avg_commute = float(np.mean(commute_list)) if commute_list else 0.0
        avg_wait = float(np.mean(wait_samples)) if wait_samples else 0.0
        return avg_commute, avg_wait
    finally:
        safe_traci_close()

def run_sotl(ep_seed=None, mu=SOTL_MU, nu=SOTL_NU, psi=SOTL_PSI, omega=SOTL_OMEGA):
    """
    SOTL controller (simplified, per paper-style).
    - Keep current phase at least MIN_GREEN_TIME seconds.
    - Accumulate 'chi' from vehicles in non-authorized lanes within psi. If chi > mu and eta==0 or eta>nu => switch.
    """
    try:
        seed_arg = ["--seed", str(ep_seed)] if ep_seed is not None else None
        start_sumo(seed_arg)
        depart_times = {}
        arrive_times = {}
        wait_samples = []
        chi_holder = {'chi': 0.0}

        # ensure a known TLS logic
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

        # run until no vehicles
        while traci.simulation.getTime() < MAX_SIM_SECONDS and traci.simulation.getMinExpectedNumber() > 0:
            # enforce min green
            t_elapsed = 0
            while t_elapsed < MIN_GREEN_TIME and traci.simulation.getMinExpectedNumber() > 0:
                t = traci.simulation.getTime()
                for vid in traci.simulation.getDepartedIDList():
                    depart_times.setdefault(vid, t)
                for vid in traci.simulation.getArrivedIDList():
                    arrive_times.setdefault(vid, t)
                wait_samples.append(compute_avg_wait_per_vehicle())
                traci.simulationStep()
                t_elapsed += 1

            # after MIN_GREEN, compute chi (vehicles in non-authorized lanes within psi)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
            phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            state_str = logic.phases[phase_idx].state
            controlled = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
            # compute per-lane authorized status from state_str mapping over controlled lanes
            authorized_set = set()
            for idx, lane in enumerate(controlled):
                if idx < len(state_str) and state_str[idx] in ('G','g'):
                    authorized_set.add(lane)
            # compute chi
            chi = 0.0
            for lane in controlled:
                if lane not in authorized_set:
                    try:
                        for vid in traci.lane.getLastStepVehicleIDs(lane):
                            # distance to stop line
                            lp = traci.vehicle.getLanePosition(vid)
                            l_len = traci.lane.getLength(lane)
                            dist_to_stop = l_len - lp
                            if dist_to_stop <= psi:
                                chi += 1.0
                    except Exception:
                        pass
            chi_holder['chi'] += chi

            # compute eta on authorized lanes within omega
            if chi_holder['chi'] > mu:
                eta = 0
                for lane in authorized_set:
                    try:
                        for vid in traci.lane.getLastStepVehicleIDs(lane):
                            lp = traci.vehicle.getLanePosition(vid)
                            l_len = traci.lane.getLength(lane)
                            dist_to_stop = l_len - lp
                            if dist_to_stop <= omega:
                                eta += 1
                    except Exception:
                        pass
                if eta == 0 or eta > nu:
                    # switch to next phase (simple cycle)
                    num_phases = len(logic.phases)
                    next_phase = (phase_idx + 1) % num_phases
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
                    chi_holder['chi'] = 0.0

        commute_list = [arrive_times[v] - depart_times[v] for v in depart_times if v in arrive_times]
        avg_commute = float(np.mean(commute_list)) if commute_list else 0.0
        avg_wait = float(np.mean(wait_samples)) if wait_samples else 0.0
        return avg_commute, avg_wait
    finally:
        safe_traci_close()

def run_maxpressure(ep_seed=None):
    """
    Full MaxPressure: for each candidate phase compute:
       pressure = sum_{links allowed in phase} queue(fromLane) - sum_{links allowed in phase} queue(toLane)
    Choose phase with maximum pressure, run MIN_GREEN_TIME, then re-evaluate.
    """
    try:
        seed_arg = ["--seed", str(ep_seed)] if ep_seed is not None else None
        start_sumo(seed_arg)
        depart_times = {}
        arrive_times = {}
        wait_samples = []

        # prepare TLS info
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        phases = logic.phases
        # controlled links: list of lists (one entry per signal index)
        ctrl_links = traci.trafficlight.getControlledLinks(TRAFFIC_LIGHT_ID)  # a list of lists of (inLane, outLane, via)
        # ctrl_lanes: flatten lanes controlled in order
        ctrl_lanes_flat = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)

        # run until no vehicles
        while traci.simulation.getTime() < MAX_SIM_SECONDS and traci.simulation.getMinExpectedNumber() > 0:
            # compute pressure per phase
            best_phase = 0
            best_pressure = -1e9
            for p_idx, p in enumerate(phases):
                state_str = p.state
                # We expect state_str length equals number of signal indices
                pressure = 0.0
                for sig_idx, links_for_sig in enumerate(ctrl_links):
                    # if this signal index is green in this phase, include its links
                    allowed = (sig_idx < len(state_str) and state_str[sig_idx] in ('G','g'))
                    if allowed:
                        # links_for_sig is list of (inLane, outLane, via)
                        for (inLane, outLane, _) in links_for_sig:
                            try:
                                q_in = traci.lane.getLastStepVehicleNumber(inLane)
                            except Exception:
                                q_in = 0
                            try:
                                q_out = traci.lane.getLastStepVehicleNumber(outLane)
                            except Exception:
                                q_out = 0
                            pressure += float(q_in - q_out)
                if pressure > best_pressure:
                    best_pressure = pressure
                    best_phase = p_idx

            # set best phase and run MIN_GREEN_TIME seconds
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, int(best_phase))
            for _ in range(MIN_GREEN_TIME):
                t = traci.simulation.getTime()
                for vid in traci.simulation.getDepartedIDList():
                    depart_times.setdefault(vid, t)
                for vid in traci.simulation.getArrivedIDList():
                    arrive_times.setdefault(vid, t)
                wait_samples.append(compute_avg_wait_per_vehicle())
                traci.simulationStep()
                if traci.simulation.getMinExpectedNumber() == 0:
                    break

        commute_list = [arrive_times[v] - depart_times[v] for v in depart_times if v in arrive_times]
        avg_commute = float(np.mean(commute_list)) if commute_list else 0.0
        avg_wait = float(np.mean(wait_samples)) if wait_samples else 0.0
        return avg_commute, avg_wait
    finally:
        safe_traci_close()

# -------------------- RL AGENT (Double DQN) --------------------
# Simple MLP agent using the same state representation as earlier: [queues..., veh_nums..., curr_phase, phase_timer]
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class MLQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.policy = MLQNetwork(state_dim, action_dim).to(device)
        self.target = MLQNetwork(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=LR)
        self.replay = ReplayBuffer(MEMORY_SIZE)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.learn_steps = 0
        self.total_env_steps = 0
        self.epsilon = EPS_START
        self.epsilon_min = EPS_END
        self.epsilon_decay = (EPS_END / EPS_START) ** (1.0 / max(1, EPS_DECAY_EPISODES))

    def act(self, state_np):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(s)
        return int(torch.argmax(q.cpu()).item())

    def push(self, *args):
        self.replay.push(*args)

    def learn(self):
        if len(self.replay) < max(REPLAY_START_SIZE, self.batch_size):
            return
        batch = self.replay.sample(self.batch_size)
        states = torch.from_numpy(np.vstack(batch.state)).float().to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.vstack(batch.next_state)).float().to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.policy(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.policy(next_states), dim=1, keepdim=True)
            q_next = self.target(next_states).gather(1, next_actions)
            q_target = rewards + (1.0 - dones) * (self.gamma * q_next)

        loss = nn.MSELoss()(q_vals, q_target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
        self.optim.step()

        # soft update target (polyak)
        tau = 0.01
        for p, tp in zip(self.policy.parameters(), self.target.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        self.learn_steps += 1

    def update_epsilon_episode_end(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

# -------------------- RL EPISODE RUN --------------------
def run_rl_episode(agent=None, ep_seed=None, train_mode=False):
    """
    Run one episode with RL agent. If train_mode True, agent learns online.
    Returns avg_commute, avg_wait, and total_env_steps consumed.
    """
    try:
        seed_arg = ["--seed", str(ep_seed)] if ep_seed is not None else None
        start_sumo(seed_arg)
        # initialize
        depart_times = {}
        arrive_times = {}
        wait_samples = []
        state_timer = 0

        # initial state vector
        queues = [traci.lane.getLastStepHaltingNumber(l) for l in INCOMING_LANES]
        veh_nums = [traci.lane.getLastStepVehicleNumber(l) for l in INCOMING_LANES]
        curr_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        state = np.array(queues + veh_nums + [curr_phase, state_timer], dtype=np.float32)

        # pick initial action
        action = agent.act(state)
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, int(action))
        steps = 0

        while traci.simulation.getTime() < MAX_SIM_SECONDS and traci.simulation.getMinExpectedNumber() > 0:
            # hold phase for MIN_GREEN_TIME
            arrived_in_window = 0
            for _ in range(MIN_GREEN_TIME):
                t = traci.simulation.getTime()
                for vid in traci.simulation.getDepartedIDList():
                    depart_times.setdefault(vid, t)
                arrived = traci.simulation.getArrivedIDList()
                for vid in arrived:
                    arrive_times.setdefault(vid, t)
                arrived_in_window += len(arrived)

                wait_samples.append(compute_avg_wait_per_vehicle())
                traci.simulationStep()
                steps += 1
                agent.total_env_steps = getattr(agent, 'total_env_steps', 0) + 1

                if traci.simulation.getMinExpectedNumber() == 0:
                    break

            # next state
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in INCOMING_LANES]
            veh_nums = [traci.lane.getLastStepVehicleNumber(l) for l in INCOMING_LANES]
            curr_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            state_timer = MIN_GREEN_TIME  # we used a fixed step window
            next_state = np.array(queues + veh_nums + [curr_phase, state_timer], dtype=np.float32)

            # reward: prioritize arrivals (clear vehicles) and penalize waits/queues
            avg_wait = compute_avg_wait_per_vehicle()
            queue_len = sum(traci.lane.getLastStepHaltingNumber(l) for l in INCOMING_LANES)
            reward = (arrived_in_window * 2.0) - (avg_wait + 0.1 * queue_len)

            done = (traci.simulation.getMinExpectedNumber() == 0)

            # store transition and learn
            agent.push(state, int(action), float(reward), next_state, float(done))
            if train_mode:
                # learn multiple times per step depending on buffer
                agent.learn()

            # pick next action
            state = next_state
            action = agent.act(state)
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, int(action))

            if done:
                break

        commute_list = [arrive_times[v] - depart_times[v] for v in depart_times if v in arrive_times]
        avg_commute = float(np.mean(commute_list)) if commute_list else 0.0
        avg_wait = float(np.mean(wait_samples)) if wait_samples else 0.0

        return avg_commute, avg_wait, steps
    finally:
        safe_traci_close()

# -------------------- EXPERIMENT DRIVER --------------------
def evaluate_all_controllers(episodes=EPISODES):
    """
    For reproducibility we choose seeds per episode. Each controller is run EPISODES times using the same seeds.
    Returns dictionaries of mean commute and mean wait for each controller.
    """
    seeds = [SEED + i for i in range(episodes)]
    fixed_commutes, fixed_waits = [], []
    sotl_commutes, sotl_waits = [], []
    mp_commutes, mp_waits = [], []
    rl_commutes, rl_waits = [], []

    # Train RL agent for a while across episodes (we do online training with agent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine action space size
    start_sumo()
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
    n_actions = len(logic.phases)
    safe_traci_close()

    rl_agent = RLAgent(STATE_SIZE, n_actions, device)

    # Optionally do some training episodes first (here we interleave: train RL for EPISODES and also evaluate rules)
    for i, s in enumerate(seeds):
        print(f"\n=== Episode {i+1}/{episodes} | seed={s} ===")

        # 1) Fixed-time
        avg_c, avg_w = run_fixed_time(ep_seed=s)
        fixed_commutes.append(avg_c); fixed_waits.append(avg_w)
        print(f"[Fixed] commute={avg_c:.2f}s wait={avg_w:.3f}s")

        # 2) SOTL
        avg_c, avg_w = run_sotl(ep_seed=s)
        sotl_commutes.append(avg_c); sotl_waits.append(avg_w)
        print(f"[SOTL] commute={avg_c:.2f}s wait={avg_w:.3f}s")

        # 3) MaxPressure
        avg_c, avg_w = run_maxpressure(ep_seed=s)
        mp_commutes.append(avg_c); mp_waits.append(avg_w)
        print(f"[MaxPressure] commute={avg_c:.2f}s wait={avg_w:.3f}s")

        # 4) RL (train_mode True: agent learns online from its actions)
        avg_c, avg_w, steps = run_rl_episode(agent=rl_agent, ep_seed=s, train_mode=True)
        rl_commutes.append(avg_c); rl_waits.append(avg_w)
        print(f"[RL] commute={avg_c:.2f}s wait={avg_w:.3f}s steps={steps} eps={rl_agent.epsilon:.4f} replay={len(rl_agent.replay)}")

        # end-of-episode updates
        rl_agent.update_epsilon_episode_end()

    # take averages
    results = {
        "Fixed": (np.mean(fixed_commutes), np.mean(fixed_waits)),
        "SOTL": (np.mean(sotl_commutes), np.mean(sotl_waits)),
        "MaxPressure": (np.mean(mp_commutes), np.mean(mp_waits)),
        "RL": (np.mean(rl_commutes), np.mean(rl_waits))
    }
    return results

# -------------------- PLOTTING --------------------
def plot_comparison(results_dict):
    labels = list(results_dict.keys())
    commutes = [results_dict[k][0] for k in labels]
    waits = [results_dict[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.6

    plt.figure(figsize=(9,5))
    plt.bar(x, commutes, width, color=['#888888','#1f77b4','#2ca02c','#d62728'])
    plt.xticks(x, labels)
    plt.ylabel("Avg commute time (s)")
    plt.title("Controller comparison — Avg commute time")
    for i, v in enumerate(commutes):
        plt.text(i, v + max(commutes)*0.01, f"{v:.1f}", ha='center')
    outpath = os.path.join(LOG_DIR, "compare_commute.png")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)

    plt.figure(figsize=(9,5))
    plt.bar(x, waits, width, color=['#888888','#1f77b4','#2ca02c','#d62728'])
    plt.xticks(x, labels)
    plt.ylabel("Avg wait per vehicle (s)")
    plt.title("Controller comparison — Avg wait")
    for i, v in enumerate(waits):
        plt.text(i, v + max(waits)*0.01, f"{v:.2f}", ha='center')
    outpath2 = os.path.join(LOG_DIR, "compare_wait.png")
    plt.tight_layout()
    plt.savefig(outpath2)
    plt.close()
    print("Saved", outpath2)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    t0 = time.time()
    print("Starting 4-way controller comparison. This will run each controller for", EPISODES, "episodes (seeds).")
    results = evaluate_all_controllers(EPISODES)
    print("\nAggregate results (mean across episodes):")
    for k, (comm, wt) in results.items():
        print(f"  {k:12s} -> Avg commute: {comm:.2f}s, Avg wait: {wt:.3f}s")
    plot_comparison(results)
    print("Done. Total elapsed:", time.time() - t0, "seconds")