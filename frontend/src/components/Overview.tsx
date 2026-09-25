import { useState } from 'react'
import type { MetricsResponse } from '../api/types'
import { decimal, integer, label, languageName, percent } from '../utils/format'
import { RatingDistributionChart, SentimentDistributionChart, AspectSentimentChart, TopIssuesChart, IssueImpactScatterChart, FeatureRequestsChart, ReviewsByCountryChart, LanguageDistributionChart, AverageRatingByCountryChart, RatingByVersionChart, IssuesOverTimeChart } from './charts'

type Props = { metrics?: MetricsResponse; loading: boolean; error: Error | null; onIssueClick: (id: string) => void }

function Section({ number, title, description, children }: { number: string; title: string; description?: string; children: React.ReactNode }) {
  return <section className="mt-16"><div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-ink pb-4"><div className="flex items-baseline gap-4"><span className="font-mono text-xs text-orange">{number}</span><h2 className="section-heading">{title}</h2></div>{description && <p className="max-w-[360px] text-xs leading-relaxed text-muted">{description}</p>}</div>{children}</section>
}
function ChartPanel({ title, detail, children, className = '' }: { title: string; detail?: string; children: React.ReactNode; className?: string }) {
  return <div className={`panel min-w-0 p-5 md:p-6 ${className}`}><div className="mb-5 flex min-h-9 items-start justify-between gap-4"><h3 className="text-[15px] font-bold">{title}</h3>{detail && <span className="max-w-[190px] text-right text-xs leading-snug text-muted">{detail}</span>}</div>{children}</div>
}
type Row = { review_count: number; average_rating: number | null; country?: string | null; language?: string | null; app_version?: string | null }
function DetailTable({ title, columns, rows }: { title: string; columns: { key: string; label: string; format?: (value: string | number | null) => string }[]; rows: Row[] }) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: columns.find(c => c.key === 'review_count')?.key ?? columns[0].key, dir: -1 })
  const value = (row: Row, key: string) => row[key as keyof Row] ?? null
  const sorted = [...rows].sort((a, b) => { const av = value(a, sort.key), bv = value(b, sort.key); if (av == null) return 1; if (bv == null) return -1; return (typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv))) * sort.dir })
  return <div className="panel min-w-0 p-5 md:p-6"><h3 className="mb-4 text-[15px] font-bold">{title}</h3>{rows.length ? <div className="overflow-x-auto"><table className="data-table"><thead><tr>{columns.map(c => <th key={c.key}><button className="text-left hover:text-ink" onClick={() => setSort(s => ({ key: c.key, dir: s.key === c.key ? (s.dir === 1 ? -1 : 1) : -1 }))}>{c.label} {sort.key === c.key ? (sort.dir === 1 ? '↑' : '↓') : ''}</button></th>)}</tr></thead><tbody>{sorted.map((row, index) => <tr key={index}>{columns.map(c => <td key={c.key} className={typeof value(row, c.key) === 'number' ? 'font-mono text-xs tabular' : ''}>{c.format ? c.format(value(row, c.key)) : String(value(row, c.key) ?? 'Unknown')}</td>)}</tr>)}</tbody></table></div> : <p className="py-8 text-sm text-muted">Not enough data</p>}</div>
}
const groupColumns = (name: string) => [
  { key: name, label: label(name), format: (v: string | number | null) => v == null ? 'Unknown' : name === 'language' ? languageName(String(v)) : name === 'country' ? String(v).toUpperCase() : String(v) },
  { key: 'review_count', label: 'Reviews', format: (v: string | number | null) => integer(v as number | null) },
  { key: 'average_rating', label: 'Avg rating', format: (v: string | number | null) => decimal(v as number | null) },
]

function Keywords({ metrics }: { metrics: MetricsResponse }) {
  const data = metrics.common_keywords_by_language ?? {}
  const languages = Object.keys(data).filter(key => data[key]?.length)
  const [selected, setSelected] = useState<string | null>(null)
  const current = selected && languages.includes(selected) ? selected : languages[0]
  const rows = current ? data[current].slice(0, 12) : []
  return <div className="panel p-5 md:p-6"><div className="mb-5 flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">Language signal</p><h3 className="mt-2 text-[15px] font-bold">Common negative keywords</h3></div>{languages.length > 0 && <select className="field !w-auto min-w-[150px]" aria-label="Keyword language" value={current} onChange={e => setSelected(e.target.value)}>{languages.map(lang => <option key={lang} value={lang}>{languageName(lang)}</option>)}</select>}</div>{rows.length ? <div className="space-y-0">{rows.map((row, i) => <div key={row.normalized_phrase || row.phrase} className="grid grid-cols-[28px_minmax(0,1fr)_auto] items-center gap-3 border-t border-line py-3 text-sm"><span className="font-mono text-xs text-orange">{String(i + 1).padStart(2, '0')}</span><span className="truncate">{row.phrase}</span><span className="font-mono text-xs tabular text-muted">{integer(row.review_count)}</span></div>)}</div> : <p className="py-10 text-sm text-muted">Not enough data</p>}<p className="mt-4 text-xs leading-relaxed text-muted">Phrases extracted from reviews classified as negative. Counts describe this language sample.</p></div>
}

export default function Overview({ metrics, loading, error, onIssueClick }: Props) {
  if (loading) return <div className="py-16 text-sm text-muted">Loading analytical metrics…</div>
  if (error) return <div role="alert" className="panel border-danger p-6 text-danger">Could not load metrics: {error.message}</div>
  if (!metrics) return <div className="panel p-6 text-muted">Metrics are not available yet.</div>
  const { basic, nlp, sentiment } = metrics
  const issueCount = nlp?.issue_metrics.issues.length
  const kpis = [
    { value: integer(basic.review_count), name: 'Reviews' },
    { value: basic.average_rating == null ? 'No data' : `${decimal(basic.average_rating)} ★`, name: 'Avg rating' },
    { value: percent(basic.negative_rating_share), name: '1–2★ reviews' },
    { value: issueCount == null ? 'No data' : integer(issueCount), name: 'Canonical issues' },
  ]
  return <div><div className="grid grid-cols-2 border-y border-ink md:grid-cols-4">{kpis.map((kpi, i) => <div key={kpi.name} className={`py-6 ${i % 2 ? 'pl-5' : ''} md:pl-7 ${i > 0 ? 'md:border-l' : ''} border-line`}><strong className="block whitespace-nowrap font-mono text-[clamp(24px,3vw,43px)] font-medium tracking-[-0.06em] tabular">{kpi.value}</strong><span className="eyebrow mt-2 block">{kpi.name}</span></div>)}</div>
    {(nlp?.analysis_summary.analysis_error_count ?? 0) > 0 && <p className="mt-5 border-l-2 border-ochre bg-sand/50 px-4 py-3 text-sm text-muted"><strong className="text-ink">{integer(nlp?.analysis_summary.analyzed_review_count)} of {integer(nlp?.analysis_summary.input_review_count)} reviews</strong> were successfully analyzed for issues and feature requests; {integer(nlp?.analysis_summary.analysis_error_count)} had extraction errors. Their shares use the successfully analyzed sample.</p>}
    <Section number="01" title="The sample at a glance" description="Rating and sentiment describe different signals. Ratings come directly from reviews; sentiment comes from text analysis."><div className="grid gap-5 lg:grid-cols-2"><ChartPanel title="Rating distribution" detail="Review count · share on hover"><RatingDistributionChart metrics={metrics} /></ChartPanel><ChartPanel title="Sentiment distribution" detail={`${integer(sentiment.sample_size)} reviews analyzed`}><SentimentDistributionChart metrics={metrics} /></ChartPanel></div></Section>
    <Section number="02" title="Where friction concentrates" description="Issues are normalized themes. Bubble size reflects the share of 1–2★ ratings among reviews mentioning each issue."><div className="grid gap-5 lg:grid-cols-2"><ChartPanel title="Issue impact map" detail="More frequent → · Lower rating ↓"><IssueImpactScatterChart metrics={metrics} onIssueClick={onIssueClick} /></ChartPanel><ChartPanel title="Top issues" detail="Click a bar for evidence"><TopIssuesChart metrics={metrics} onIssueClick={onIssueClick} /></ChartPanel><ChartPanel title="Aspect sentiment" detail="Mixed sentiment can occur in one review" className="lg:col-span-2"><AspectSentimentChart metrics={metrics} /></ChartPanel></div></Section>
    <Section number="03" title="What users ask for"><div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]"><ChartPanel title="Feature requests" detail="Distinct reviews mentioning each request"><FeatureRequestsChart metrics={metrics} /></ChartPanel><Keywords metrics={metrics} /></div></Section>
    <Section number="04" title="Markets & languages" description="Country counts may overlap when a review was observed in multiple storefronts."><div className="grid gap-5 lg:grid-cols-2"><ChartPanel title="Reviews by country"><ReviewsByCountryChart metrics={metrics} /></ChartPanel><ChartPanel title="Average rating by country" detail="Sample size shown on hover"><AverageRatingByCountryChart metrics={metrics} /></ChartPanel><ChartPanel title="Language distribution"><LanguageDistributionChart metrics={metrics} /></ChartPanel><DetailTable title="Country details" columns={groupColumns('country')} rows={basic.country_statistics} /><DetailTable title="Language details" columns={groupColumns('language')} rows={basic.language_statistics} /></div></Section>
    <Section number="05" title="Version & time" description="Patterns describe collected reviews only; small groups should be read with care."><div className="grid gap-5 lg:grid-cols-2"><ChartPanel title="Rating by app version"><RatingByVersionChart metrics={metrics} /></ChartPanel><DetailTable title="Version details" columns={groupColumns('app_version')} rows={basic.rating_by_version} /><ChartPanel title="Issues over time" detail="Only shown when multiple periods are available" className="lg:col-span-2"><IssuesOverTimeChart metrics={metrics} /></ChartPanel></div></Section>
    <p className="mt-10 border-t border-line pt-4 text-xs text-muted">Based on the collected App Store review sample. Issue and feature request shares use successfully analyzed reviews as their denominator.</p>
  </div>
}
