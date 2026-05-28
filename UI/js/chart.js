/* ============================================================
   Chart Manager — Chart.js integration
   ============================================================ */

const ChartManager = (() => {
  const charts = {};

  const COLORS = {
    correct: '#22c55e',
    incorrect: '#ef4444',
    correctBg: 'rgba(34, 197, 94, 0.2)',
    incorrectBg: 'rgba(239, 68, 68, 0.2)',
    primary: '#4f6ef7',
    primaryBg: 'rgba(79, 110, 247, 0.2)',
    warning: '#f59e0b',
  };

  function destroy(id) {
    if (charts[id]) {
      charts[id].destroy();
      delete charts[id];
    }
  }

  function createPie(id, correct, incorrect) {
    destroy(id);
    const canvas = document.getElementById(id);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    charts[id] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Correct', 'Incorrect'],
        datasets: [{
          data: [correct, incorrect],
          backgroundColor: [COLORS.correct, COLORS.incorrect],
          borderWidth: 0,
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { padding: 16, usePointStyle: true, font: { family: 'Inter', size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
                return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
              },
            },
          },
        },
        cutout: '65%',
      },
    });
  }

  function createBar(id, labels, values, label) {
    destroy(id);
    const canvas = document.getElementById(id);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const barColors = values.map(v => v >= 0.5 ? COLORS.correct : COLORS.incorrect);
    const barBg = values.map(v => v >= 0.5 ? COLORS.correctBg : COLORS.incorrectBg);

    charts[id] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: label || 'Confidence',
          data: values,
          backgroundColor: barColors,
          borderColor: barColors,
          borderWidth: 0,
          borderRadius: 4,
          barThickness: 18,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${(ctx.parsed.x * 100).toFixed(0)}%`,
            },
          },
        },
        scales: {
          x: {
            min: 0,
            max: 1,
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: {
              callback: v => `${(v * 100).toFixed(0)}%`,
              font: { size: 10 },
            },
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 10 } },
          },
        },
      },
    });
  }

  return {
    renderPie(id, correct, incorrect) { createPie(id, correct, incorrect); },
    renderBar(id, labels, values, label) { createBar(id, labels, values, label); },
    destroy(id) { destroy(id); },
    destroyAll() { Object.keys(charts).forEach(destroy); },
  };
})();
