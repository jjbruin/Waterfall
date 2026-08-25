/**
 * ECharts option builders for the Portfolio Snapshot page-1 charts.
 *
 * Separated from the .vue components so the option objects can be exercised
 * headlessly against real backend payloads — a chart whose config is only
 * reachable inside a component can only be checked by eye, and by then the
 * numbers are pixels. `scripts/snapshot_summary_charts_check.mjs` imports THESE
 * functions, so what it asserts is what the page renders.
 *
 * Colour, mark and label rules live in ./palette.ts.
 */
import { hueFor, SURFACE, SEGMENT_GAP, INK_AXIS, INK_GRID } from './palette'

/** Below this share of its own bar, a stacked segment gets no inside label. */
export const LABEL_MIN_PCT = 5

/**
 * Below this share, a pie slice gets no inside percentage.
 *
 * Lower than the bar's threshold because a slice near the rim has more room
 * than a 3%-tall bar segment, but not zero: City West arrives as a 1.2%
 * "Unclassified" sliver at 26Q1, and "1%" printed on a wedge two degrees wide
 * lands on top of its neighbour's label. The legend, the tooltip and the
 * "Show data" table all still carry it.
 */
export const PIE_LABEL_MIN_PCT = 3

export interface Bucket {
  label: string
  funded?: number | null
  committed?: number | null
  funded_pct?: number | null
  committed_pct?: number | null
  deal_count?: number | null
}

export interface Alloc {
  buckets?: Bucket[]
  total_funded?: number | null
  total_committed?: number | null
  deal_count?: number | null
}

export const BAR_NAMES = ['Currently Funded', 'Total Commitment'] as const

export function m$(v: number | null | undefined): string {
  if (v == null) return '—'
  return '$' + (v / 1e6).toFixed(1) + 'M'
}

export function dollars(v: number | null | undefined): string {
  if (v == null) return '—'
  return '$' + Math.round(v).toLocaleString('en-US')
}

/**
 * Asset Allocation — two 100%-stacked bars, one per basis.
 *
 * The series values ARE percentages of each bar's own total: ECharts does not
 * normalise a stack, and the page's question is about mix, not size. The dollar
 * rides along on each datum so the label formatter never recomputes it.
 */
export function assetAllocationOption(alloc: Alloc | null | undefined) {
  const buckets = alloc?.buckets || []
  const tf = alloc?.total_funded || 0
  const tc = alloc?.total_committed || 0
  const df = tf || 1
  const dc = tc || 1

  const series = buckets.map((b, i) => {
    const pts = [
      { pct: ((b.funded || 0) / df) * 100, usd: b.funded ?? null },
      { pct: ((b.committed || 0) / dc) * 100, usd: b.committed ?? null },
    ]
    return {
      name: b.label,
      type: 'bar',
      stack: 'alloc',
      barWidth: '46%',
      itemStyle: {
        color: hueFor(i, b.label),
        borderColor: SURFACE,
        borderWidth: SEGMENT_GAP,
      },
      label: {
        show: true,
        position: 'inside',
        color: '#ffffff',
        fontSize: 10,
        fontWeight: 600,
        formatter: (p: any) => {
          const d = pts[p.dataIndex]
          return d && d.pct >= LABEL_MIN_PCT ? dollars(d.usd) : ''
        },
      },
      data: pts.map((d) => ({ value: d.pct, usd: d.usd })),
    }
  })

  return {
    grid: { left: 52, right: 16, top: 14, bottom: 64, containLabel: false },
    tooltip: {
      trigger: 'item',
      formatter: (p: any) =>
        `<b>${p.seriesName}</b><br/>${BAR_NAMES[p.dataIndex]}: `
        + `${dollars(p.data?.usd)} (${p.value.toFixed(1)}%)`,
    },
    legend: {
      bottom: 0,
      itemWidth: 11,
      itemHeight: 11,
      icon: 'roundRect',
      textStyle: { fontSize: 11, color: INK_AXIS },
    },
    xAxis: {
      type: 'category',
      data: [`${BAR_NAMES[0]}\n${m$(tf)}`, `${BAR_NAMES[1]}\n${m$(tc)}`],
      axisLine: { lineStyle: { color: INK_GRID } },
      axisTick: { show: false },
      axisLabel: { fontSize: 11, lineHeight: 16, color: INK_AXIS },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      interval: 20,
      axisLabel: { formatter: '{value}%', fontSize: 10, color: INK_AXIS },
      splitLine: { lineStyle: { color: INK_GRID } },
    },
    series,
  }
}

/**
 * Deal Type — a three-slice pie on FUNDED dollars.
 *
 * Funded, not committed: the published slices sum to the funded total and the
 * narrative above quotes funded figures, so committed would answer a different
 * question with the same picture.
 */
export function dealTypeOption(alloc: Alloc | null | undefined) {
  const buckets = alloc?.buckets || []
  return {
    tooltip: {
      trigger: 'item',
      formatter: (p: any) =>
        `<b>${p.name}</b><br/>${dollars(p.value)} (${p.percent.toFixed(1)}%)`
        + `<br/><span style="color:#6c757d">${p.data?.deals ?? 0} deal(s)</span>`,
    },
    legend: {
      bottom: 0,
      itemWidth: 11,
      itemHeight: 11,
      icon: 'roundRect',
      textStyle: { fontSize: 11, color: INK_AXIS },
    },
    series: [
      {
        type: 'pie',
        radius: '66%',
        center: ['50%', '46%'],
        itemStyle: { borderColor: SURFACE, borderWidth: SEGMENT_GAP },
        label: {
          show: true,
          position: 'inside',
          color: '#ffffff',
          fontSize: 13,
          fontWeight: 700,
          // Selective, like the bar: a sliver's label would sit on its
          // neighbour's. See PIE_LABEL_MIN_PCT.
          formatter: (p: any) =>
            p.percent >= PIE_LABEL_MIN_PCT ? `${p.percent.toFixed(0)}%` : '',
        },
        labelLine: { show: false },
        emphasis: { scale: true, scaleSize: 4 },
        data: buckets.map((b, i) => ({
          name: b.label,
          value: b.funded || 0,
          deals: b.deal_count,
          itemStyle: { color: hueFor(i, b.label) },
        })),
      },
    ],
  }
}
