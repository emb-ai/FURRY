"""Render only the 22 published samples. Requires matplotlib; no raw recordings."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with (HERE / 'low-transfer-22.csv').open() as stream:
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]

def column(name, scale=1):
    return [row[name] * scale for row in rows]

x = column('source_elapsed_s')
end = rows[-1]
dx, dz = end['raw_ball_left_dx_m'], end['raw_ball_left_dz_m']
norm = (dx * dx + dz * dz) ** .5
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(4, 1, figsize=(10.5, 10), sharex=True)
fig.suptitle('Низкий референс и фактический перенос — разные величины', fontsize=15)
fig.text(.10, .936, 'Meta84 · episode 1791652646384286 · 22 уникальных source samples', fontsize=10)
for side, label, color in [('left', 'Левая ball', '#1565c0'), ('right', 'Правая ball', '#d87519')]:
    projected = [(row[f'raw_ball_{side}_dx_m'] * dx + row[f'raw_ball_{side}_dz_m'] * dz) / norm * 100 for row in rows]
    axes[0].plot(x, projected, '.-', label=label, color=color)
axes[0].set_ylabel('Meta: горизонтальный\nперенос, см')
axes[0].text(.02, .65, 'Левая ball: 48.7 см\nПравая ball: 1.29 см', transform=axes[0].transAxes)
for name, label, color in [('raw_ankle_left_y_m', 'Meta ankle', '#1565c0'), ('raw_ball_left_y_m', 'Meta ball', '#008577')]:
    axes[1].plot(x, column(name, 1000), '.-', label=label, color=color)
axes[1].set_ylabel('Meta: высота левой\nанатомической точки, мм')
for name, label, color in [('legacy_reference_sole_left_m', 'Исходный GMR', '#00796b'), ('root_morph_reference_sole_left_m', 'Root-morph: прототип', '#9c27b0')]:
    axes[2].plot(x, column(name, 1000), '.-', label=label, color=color)
axes[2].axhline(35, ls='--', color='#bf3535', label='Вход SWING: 35 мм / 40 мс')
axes[2].set_ylabel('GMR FK: минимум\nвсей левой подошвы, мм')
for name, label, color in [('legacy_physical_sole_left_m', 'Исходный mapping', '#00796b'), ('root_morph_physical_sole_left_m', 'Root-morph: прототип', '#9c27b0')]:
    axes[3].plot(x, column(name, 1000), '.-', label=label, color=color)
axes[3].axhline(0, lw=1, color='#626873')
axes[3].set_ylabel('Физический робот:\nминимум подошвы, мм')
axes[3].set_xlabel('Время источника, с; физика сопоставлена по recorded_sim_time_s')
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(loc='upper right', fontsize=9)
fig.text(.10, .015, 'Офлайн A/B, TWIST2 +500 / 50 Гц; одинаковое начальное состояние, собственная обратная связь.\nMeta ball ≠ подошва G1. Root-morph не принят: качество полных прогонов смешанное.', fontsize=9)
fig.subplots_adjust(top=.90, bottom=.10, left=.13, right=.98, hspace=.20)
fig.savefig(HERE / 'low-transfer.png', dpi=130)
