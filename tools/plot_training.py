"""Regenerate the scientific training figure from archived CSVs. Needs matplotlib."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = ROOT / 'results/training/markdown-v3'
    with (source / 'losses-complete.csv').open(encoding='utf-8') as f:
        train = list(csv.DictReader(f))
    with (source / 'evaluation-complete.csv').open(encoding='utf-8') as f:
        ev = list(csv.DictReader(f))
    x = [int(r['phase_step']) for r in train]
    y = [float(r['train_loss']) for r in train]
    mean = [sum(y[max(0, i-24):i+1]) / min(i+1, 25) for i in range(len(y))]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.2), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(x, y, color='#9bb8ce', linewidth=.45, alpha=.55, label='Train : valeurs brutes par mise à jour')
    ax.plot(x, mean, color='#22678a', linewidth=1.1, label='Train : moyenne glissante de 25 mises à jour')
    ex = [int(r['phase_step']) for r in ev]
    ax.plot(ex, [float(r['eval_loss']) for r in ev], color='#d66b31', linewidth=1.6, label='Validation : mesure réelle sur 256 pages')
    ax.plot(ex, [float(r['tables_loss']) for r in ev], color='#8c68a6', linestyle='--', linewidth=.9, label='Validation tableaux (128)')
    ax.plot(ex, [float(r['control_loss']) for r in ev], color='#4f946b', linestyle='--', linewidth=.9, label='Validation contrôles (128)')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('Passe Markdown complète — 1 841 540 vues / 4 796 mises à jour', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    axes[1].plot(x, [float(r['learning_rate']) for r in train], color='#22678a')
    axes[1].set_ylabel('Learning rate')
    axes[1].set_xlabel('Pas de la phase Markdown (le compteur historique commence à 14355)')
    for a in axes:
        a.grid(alpha=.15)
        a.axvline(4682, color='#48505a', linestyle=':', linewidth=1)
    axes[1].annotate('Reprise après 4682', (4682, 1.15e-5), xytext=(3200, 1.5e-5), arrowprops={'arrowstyle': '->', 'color': '#48505a'}, fontsize=8)
    axes[1].set_xlim(0, 4850)
    fig.text(.07, .012, 'Source : logs réels relus le 30/10/2026. Loss de monitoring ≠ qualité OCR ; train et validation ont des pondérations différentes.', fontsize=8, color='#46515d')
    fig.tight_layout(rect=(0, .045, 1, 1))
    target = ROOT / 'reports/training'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'markdown-losses-complete.png', dpi=170)
    fig.savefig(target / 'markdown-losses-complete.svg')
    plt.close(fig)
    print('Complete measured training figure saved.')

if __name__ == '__main__':
    main()
