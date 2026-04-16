import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blackjack_rl.agent.q_agent import QLearningAgent
from blackjack_rl.strategy import basic_strategy

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(root_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#4a4a6a',
    'axes.labelcolor': '#e0e0e0',
    'axes.titlecolor': '#ffffff',
    'xtick.color': '#c0c0c0',
    'ytick.color': '#c0c0c0',
    'text.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'grid.linewidth': 0.6,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#4a4a6a',
    'font.family': 'DejaVu Sans',
})

BLUE   = '#4fc3f7'
ORANGE = '#ffb74d'
GREEN  = '#81c784'
RED    = '#e57373'
PURPLE = '#ce93d8'
GREY   = '#78909c'

checkpoints = np.arange(0.5, 20.5, 0.5)

td_rewards = [
    -0.1104, -0.1793,  0.0297,  0.0222,  0.1193,  0.1721,  0.1244,  0.2083,
     0.2577,  0.2849,  0.3100,  0.1640,  0.1725,  0.2576,  0.2919,  0.2510,
     0.1818,  0.3550,  0.1958,  0.3135,  0.1506,  0.2341,  0.3065,  0.3897,
     0.2636,  0.2540,  0.3250,  0.3287,  0.2342,  0.3215,  0.3358,  0.2408,
     0.3191,  0.3204,  0.4174,  0.2283,  0.2752,  0.3230,  0.1416,  0.3515,
]

mc_rewards = [
    -0.1635, -0.0388, -0.0498,  0.1422,  0.2215,  0.2921,  0.3027,  0.4016,
     0.2982,  0.5080,  0.4281,  0.3872,  0.4715,  0.3392,  0.3986,  0.4772,
     0.4744,  0.4237,  0.3739,  0.3805,  0.4965,  0.4053,  0.4284,  0.4160,
     0.5029,  0.5078,  0.3711,  0.3838,  0.5438,  0.4248,  0.4907,  0.4437,
     0.3952,  0.4546,  0.4852,  0.4264,  0.3986,  0.4260,  0.4206,  0.4135,
]

# approximate BS reward on TC+1-4 uniform sample with scaled bets — avg ~0.041
bs_level = 0.041

tc_buckets = list(range(-4, 5))

td_ev = [-90.91, -89.90, -88.80, -86.96, -68.58, -17.51,  4.18,  6.67, 11.13]
mc_ev = [-91.46, -90.67, -88.86, -86.61, -69.27, -13.60,  2.99,  9.71, 13.30]
bs_ev = [ -3.74,  -2.07,   0.39,  -3.14,  -2.13,   1.51, -0.76,  1.41,  0.99]

version_labels = ['V1\nTD flat', 'V2\nTD scaled', 'V3\nBalanced',
                  'V4\nPhase-0', 'V6 TD\nbs-anchored',
                  'V7 MC\nbs-anchored', 'Pure BS\nbenchmark']
ruin_rates  = [32.4, 81.9, 80.5, 87.9, 65.4, 64.8, 16.3]
ruin_colors = [ORANGE, RED, RED, RED, RED, ORANGE, GREEN]

brl_labels = ['Ruined\n$0', '$1–\n$499', '$500–\n$999', '$1k–\n$1.5k', '$1.5k–\n$2k', '$2k+']
bs_dist = [16.4,  9.6, 22.8, 24.7, 16.7, 9.8]
mc_dist = [64.9,  9.0, 14.0,  7.8,  2.8, 1.5]

action_labels  = ['Hit', 'Stand', 'Double', 'Split']
action_colors = ['#e57373', '#81c784', '#4fc3f7', '#ce93d8']


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Progress: TD Q-Learning vs Monte Carlo\n'
                 '(avg reward over 10k episodes at TC+1–4, scaled by bet multiplier)',
                 fontsize=13, fontweight='bold', y=1.02)

    def smooth(data, window=7):
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')

    for ax, rewards, title, color in [
        (axes[0], td_rewards, 'V6 — TD Q-Learning (20M episodes)', BLUE),
        (axes[1], mc_rewards, 'V7 — Monte Carlo (20M episodes)',    ORANGE),
    ]:
        raw = np.array(rewards)
        ax.plot(checkpoints, raw,          color=color, alpha=0.25, linewidth=1)
        ax.plot(checkpoints, smooth(raw),  color=color, linewidth=2.2, label='Agent (smoothed)')
        ax.axhline(0,        color=GREY,  linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(bs_level, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.9,
                   label=f'Basic Strategy level (~{bs_level:.3f})')
        ax.axvline(3.9, color='#ff8a65', linewidth=1.2, linestyle=':', alpha=0.7,
                   label='ε → 0.01 (~3.9M eps)')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Episodes (millions)', fontsize=10)
        ax.set_ylabel('Avg reward / 10k episodes', fontsize=9)
        ax.set_xlim(0, 20)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig1_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def fig_ev_by_bucket():
    fig, (ax_all, ax_high) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('EV/Unit Wagered by True-Count Bucket — V6 TD vs V7 MC vs Basic Strategy\n'
                 '(snapshot eval, 100k episodes each)',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(tc_buckets))
    w = 0.27

    for ax, title, xlim, ylim in [
        (ax_all,  'All TC buckets (TC −4 to +4)',   None,        None),
        (ax_high, 'High counts only (TC +1 to +4)', (3.5, 8.5), (-5, 18)),
    ]:
        ax.bar(x - w, td_ev, w, label='V6 TD Agent',    color=BLUE,   alpha=0.85)
        ax.bar(x,     mc_ev, w, label='V7 MC Agent',    color=ORANGE, alpha=0.85)
        ax.bar(x + w, bs_ev, w, label='Basic Strategy', color=GREEN,  alpha=0.85)
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels([f'TC{"+" if b > 0 else ""}{b}' for b in tc_buckets], fontsize=9)
        ax.set_ylabel('EV / unit wagered (%)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y')
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            for xi, (v6, v7, bs) in enumerate(zip(td_ev, mc_ev, bs_ev)):
                if tc_buckets[xi] >= 1:
                    for val, offset, col in [
                        (v6, -w, BLUE), (v7, 0, ORANGE), (bs, w, GREEN)
                    ]:
                        ax.text(xi + offset, val + 0.3, f'{val:+.1f}%',
                                ha='center', va='bottom', fontsize=7, color='white')
        ax.axvspan(4.5, 8.5, alpha=0.07, color=GREEN)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig2_ev_by_bucket.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def fig_bankroll():
    fig, (ax_ruin, ax_dist) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Bankroll Simulation — $1,000 start · 500 hands · 1,000 paired sessions',
                 fontsize=13, fontweight='bold')

    bars = ax_ruin.bar(range(len(version_labels)), ruin_rates,
                       color=ruin_colors, alpha=0.88, edgecolor='#ffffff22', linewidth=0.6)
    ax_ruin.axhline(16.3, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.9,
                    label='Pure BS benchmark (16.3%)')
    for bar, rate in zip(bars, ruin_rates):
        ax_ruin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=9,
                     fontweight='bold', color='white')
    ax_ruin.set_xticks(range(len(version_labels)))
    ax_ruin.set_xticklabels(version_labels, fontsize=8.5)
    ax_ruin.set_ylabel('Ruin Rate (%)', fontsize=10)
    ax_ruin.set_title('Ruin Rate by Version  (lower is better)', fontsize=11)
    ax_ruin.set_ylim(0, 100)
    ax_ruin.legend(fontsize=9)
    ax_ruin.grid(True, axis='y')

    x = np.arange(len(brl_labels))
    w = 0.27
    ax_dist.bar(x - w, bs_dist, w, label='Pure BS',             color=GREEN,  alpha=0.85)
    ax_dist.bar(x,     mc_dist, w, label='V7 MC (bs-anchored)', color=ORANGE, alpha=0.85)
    for xi, (bs, mc) in enumerate(zip(bs_dist, mc_dist)):
        ax_dist.text(xi - w, bs + 0.5, f'{bs:.1f}%', ha='center', va='bottom',
                     fontsize=7.5, color=GREEN)
        ax_dist.text(xi,     mc + 0.5, f'{mc:.1f}%', ha='center', va='bottom',
                     fontsize=7.5, color=ORANGE)
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(brl_labels, fontsize=9)
    ax_dist.set_ylabel('Sessions (%)', fontsize=10)
    ax_dist.set_title('Final Bankroll Distribution', fontsize=11)
    ax_dist.legend(fontsize=9)
    ax_dist.grid(True, axis='y')

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig3_bankroll.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def action_grid(agent, usable_ace, tc):
    totals  = list(range(4, 22))
    dealers = list(range(1, 11))
    grid = np.full((len(totals), len(dealers)), np.nan)
    for i, pt in enumerate(totals):
        for j, du in enumerate(dealers):
            can_dbl = 1.0 if pt <= 20 else 0.0
            state = np.array([pt, du, float(usable_ace), float(tc),
                              can_dbl, 0.0, 0, 3.5, 0], dtype=np.float32)
            mask  = np.array([True, True, can_dbl == 1.0, False], dtype=bool)
            grid[i, j] = agent.select_greedy(state, mask)
    return grid


def bs_grid(usable_ace):
    totals  = list(range(4, 22))
    dealers = list(range(1, 11))
    grid = np.full((len(totals), len(dealers)), np.nan)
    for i, pt in enumerate(totals):
        for j, du in enumerate(dealers):
            grid[i, j] = basic_strategy(pt, du, usable_ace, True, False)
    return grid


def draw_heatmap(ax, grid, title, cmap, norm, show_letters=True):
    dealers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    totals  = list(range(4, 22))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    ax.set_xticks(range(10));           ax.set_xticklabels(dealers, fontsize=7.5)
    ax.set_yticks(range(len(totals)));  ax.set_yticklabels(totals, fontsize=6.5)
    ax.set_xlabel('Dealer Upcard', fontsize=8)
    ax.set_ylabel('Player Total', fontsize=8)
    ax.set_title(title, fontsize=9.5, pad=5)
    if show_letters:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, action_labels[int(grid[i, j])][0],
                        ha='center', va='center', fontsize=5.5,
                        color='white', fontweight='bold')


def fig_qtable_heatmaps(agent):
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Action Heatmaps — V7 MC Agent (TC+2/+3/+4) vs Basic Strategy Reference\n'
                 'H=Hit  S=Stand  D=Double  P=Split',
                 fontsize=13, fontweight='bold', y=1.01)

    specs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.28)

    panels = [
        (0, 0, False, 2,    'Hard — Agent TC+2'),
        (0, 1, False, 3,    'Hard — Agent TC+3'),
        (0, 2, False, 4,    'Hard — Agent TC+4'),
        (0, 3, False, None, 'Hard — Basic Strategy'),
        (1, 0, True,  2,    'Soft — Agent TC+2'),
        (1, 1, True,  3,    'Soft — Agent TC+3'),
        (1, 2, True,  4,    'Soft — Agent TC+4'),
        (1, 3, True,  None, 'Soft — Basic Strategy'),
    ]

    for row, col, soft, tc, title in panels:
        ax = fig.add_subplot(specs[row, col])
        grid = bs_grid(soft) if tc is None else action_grid(agent, soft, tc)
        draw_heatmap(ax, grid, title, cmap, norm)
        if tc is None:
            for spine in ax.spines.values():
                spine.set_edgecolor(GREEN)
                spine.set_linewidth(2)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(action_colors, action_labels)]
    patches.append(mpatches.Patch(facecolor='none', edgecolor=GREEN,
                                  linewidth=2, label='Basic Strategy column'))
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=10, framealpha=0.3, bbox_to_anchor=(0.5, -0.03))

    path = os.path.join(figures_dir, 'fig4_qtable_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def fig_agent_vs_bs(agent):
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('V7 MC Agent vs Basic Strategy — Hard Hands, TC +3\n'
                 'Yellow borders = disagreements (agent departs from BS)',
                 fontsize=12, fontweight='bold')

    ag  = action_grid(agent, False, 3)
    bsg = bs_grid(False)

    draw_heatmap(axes[0], ag,  'V7 MC Agent Decisions',    cmap, norm)
    draw_heatmap(axes[1], bsg, 'Basic Strategy (reference)', cmap, norm)

    for ax in axes:
        for i in range(ag.shape[0]):
            for j in range(ag.shape[1]):
                if ag[i, j] != bsg[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor='yellow', linewidth=1.8))

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(action_colors, action_labels)]
    patches.append(mpatches.Patch(facecolor='none', edgecolor='yellow',
                                  linewidth=1.8, label='Disagreement'))
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=10, framealpha=0.3, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig5_agent_vs_bs.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def fig_qvalue_confidence(agent):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('V7 MC Agent — Decision Confidence: Q[Stand] − Q[Hit] for Hard Hands\n'
                 'Green = agent prefers Stand  |  Red = agent prefers Hit  |  Brighter = more confident',
                 fontsize=12, fontweight='bold')

    totals  = list(range(4, 22))
    dealers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    for ax, tc in zip(axes, [2, 3, 4]):
        diff = np.zeros((len(totals), 10))
        for i, pt in enumerate(totals):
            for j, du in enumerate(range(1, 11)):
                state = np.array([pt, du, 0.0, float(tc), 0.0, 0.0, 0, 3.5, 0], dtype=np.float32)
                s = agent._encode(state)
                diff[i, j] = agent.Q[s][1] - agent.Q[s][0]

        vmax = max(abs(diff).max(), 1.0)
        im = ax.imshow(diff, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                       aspect='auto', origin='lower')
        ax.set_xticks(range(10));           ax.set_xticklabels(dealers, fontsize=8)
        ax.set_yticks(range(len(totals)));  ax.set_yticklabels(totals, fontsize=7)
        ax.set_xlabel('Dealer Upcard', fontsize=9)
        ax.set_ylabel('Player Total', fontsize=9)
        ax.set_title(f'TC +{tc}', fontsize=11)
        ax.axhline(12.5, color='white', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.text(9.6, 13.2, 'BS: Stand', color='white', fontsize=7, ha='right', alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.75, label='Q[Stand]−Q[Hit]')

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig6_qvalue_confidence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def fig_algorithm_summary():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('TD Q-Learning vs Monte Carlo vs Basic Strategy\n'
                 '(snapshot eval, TC+1–4, 100k episodes)',
                 fontsize=13, fontweight='bold')

    tc_high  = [1, 2, 3, 4]
    idx      = [tc_buckets.index(t) for t in tc_high]
    v6_high  = [td_ev[i] for i in idx]
    v7_high  = [mc_ev[i] for i in idx]
    bsh      = [bs_ev[i] for i in idx]
    x        = np.arange(len(tc_high))
    w        = 0.27

    ax = axes[0]
    ax.bar(x - w, v6_high, w, label='V6 TD Agent',    color=BLUE,   alpha=0.88)
    ax.bar(x,     v7_high, w, label='V7 MC Agent',    color=ORANGE, alpha=0.88)
    ax.bar(x + w, bsh,     w, label='Basic Strategy', color=GREEN,  alpha=0.88)
    ax.axhline(0, color=GREY, linewidth=0.8, linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels([f'TC +{t}' for t in tc_high], fontsize=9)
    ax.set_ylabel('EV / unit wagered (%)', fontsize=9)
    ax.set_title('EV/Unit at High Counts', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, axis='y')
    for xi, (v6, v7, bs) in enumerate(zip(v6_high, v7_high, bsh)):
        for val, offset in [(v6, -w), (v7, 0), (bs, w)]:
            ax.text(xi + offset, val + (0.2 if val >= 0 else -1.2),
                    f'{val:+.1f}', ha='center', fontsize=7, color='white')

    ax = axes[1]
    td_gap = [v6 - bs for v6, bs in zip(v6_high, bsh)]
    mc_gap = [v7 - bs for v7, bs in zip(v7_high, bsh)]
    ax.bar(x - w/2, td_gap, w, label='V6 TD vs BS', color=BLUE,   alpha=0.88)
    ax.bar(x + w/2, mc_gap, w, label='V7 MC vs BS', color=ORANGE, alpha=0.88)
    ax.axhline(0, color=GREY, linewidth=1.2, linestyle='--', label='= Basic Strategy level')
    ax.set_xticks(x); ax.set_xticklabels([f'TC +{t}' for t in tc_high], fontsize=9)
    ax.set_ylabel('EV vs Basic Strategy (pp)', fontsize=9)
    ax.set_title('Agent EV Gap vs Basic Strategy\n(positive = beats BS)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, axis='y')
    for xi, (td, mc) in enumerate(zip(td_gap, mc_gap)):
        ax.text(xi - w/2, td + (0.2 if td >= 0 else -1.2),
                f'{td:+.1f}', ha='center', fontsize=7.5, color='white')
        ax.text(xi + w/2, mc + (0.2 if mc >= 0 else -1.2),
                f'{mc:+.1f}', ha='center', fontsize=7.5, color='white')

    ax = axes[2]
    half = len(checkpoints) // 2
    td_mean = np.mean(td_rewards[half:])
    mc_mean = np.mean(mc_rewards[half:])
    ax.plot(checkpoints[half:], td_rewards[half:], color=BLUE,   linewidth=1.5,
            marker='o', markersize=2.5, alpha=0.7, label='V6 TD')
    ax.plot(checkpoints[half:], mc_rewards[half:], color=ORANGE, linewidth=1.5,
            marker='s', markersize=2.5, alpha=0.7, label='V7 MC')
    ax.axhline(td_mean,   color=BLUE,   linewidth=1.2, linestyle='--',
               alpha=0.6, label=f'TD mean  {td_mean:.3f}')
    ax.axhline(mc_mean,   color=ORANGE, linewidth=1.2, linestyle='--',
               alpha=0.6, label=f'MC mean  {mc_mean:.3f}')
    ax.axhline(bs_level,  color=GREEN,  linewidth=1.8, linestyle='-.',
               alpha=0.9, label=f'Basic Strategy ~{bs_level:.3f}')
    ax.set_xlabel('Episodes (millions)', fontsize=9)
    ax.set_ylabel('Avg reward / 10k episodes', fontsize=9)
    ax.set_title('Converged Reward (10M–20M) vs BS Level', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig7_algorithm_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved: {path}')
    plt.close(fig)


def main():
    print('Loading V7 MC Q-table...')
    agent = QLearningAgent()
    qtable_path = os.path.join(root_dir, 'qtable_v7_mc.npy')
    if not os.path.exists(qtable_path):
        print(f'ERROR: {qtable_path} not found. Run train_v7_mc.py first.')
        sys.exit(1)
    agent.load(qtable_path)
    agent.epsilon = 0.0

    print(f'Generating figures → {figures_dir}/')
    fig_training_curves()
    fig_ev_by_bucket()
    fig_bankroll()
    fig_qtable_heatmaps(agent)
    fig_agent_vs_bs(agent)
    fig_qvalue_confidence(agent)
    fig_algorithm_summary()
    print('Done. 7 figures saved.')


if __name__ == '__main__':
    main()
