import { useMemo, useState } from 'react'
import { ArrowRight, Search } from 'lucide-react'
import type { CatalogEntry, FeatureRequestMetric, IssueMetric, IssuesResponse, MetricsResponse } from '../api/types'
import { decimal, integer, label, percent } from '../utils/format'

type Props = { metrics?: MetricsResponse; catalog?: IssuesResponse; loading: boolean; error: Error | null; selectedId?: string; onSelect: (id: string) => void; onReviewClick: (ids: string[]) => void }
type SortKey = 'canonical_name' | 'review_count' | 'share_of_successful_reviews' | 'average_rating' | 'negative_rating_share' | 'category'

export default function IssuesView({ metrics, catalog, loading, error, selectedId, onSelect, onReviewClick }: Props) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: SortKey; direction: 1 | -1 }>({ key: 'review_count', direction: -1 })
  if (loading) return <p className="py-16 text-sm text-muted">Loading issue analysis…</p>
  if (error) return <div role="alert" className="panel border-danger p-6 text-danger">Could not load issues: {error.message}</div>
  const issues = metrics?.nlp?.issue_metrics.issues ?? []
  const features = metrics?.nlp?.feature_request_metrics.feature_requests ?? []
  const active = issues.find(x => x.canonical_id === selectedId) ?? [...issues].sort((a, b) => b.review_count - a.review_count)[0]
  const entry = catalog?.issues.find(x => x.canonical_id === active?.canonical_id)
  const filtered = issues.filter(x => (x.canonical_name ?? '').toLowerCase().includes(query.toLowerCase())).sort((a, b) => b.review_count - a.review_count)
  const sorted = [...issues].sort((a, b) => {
    const av = a[sort.key], bv = b[sort.key]
    if (av == null) return 1
    if (bv == null) return -1
    return (typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv))) * sort.direction
  })
  return <div>
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Normalized signals</p><h2 className="section-heading mt-2">Issues & requests</h2></div><p className="max-w-[380px] text-sm leading-relaxed text-muted">Explore each recurring issue, its measured reach, and the original formulations behind its canonical name.</p></div>
    <div className="grid min-h-[520px] gap-5 lg:grid-cols-[350px_minmax(0,1fr)]">
      <aside className="panel flex max-h-[650px] flex-col"><div className="border-b border-line p-5"><label className="relative block"><Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" size={16} /><input className="field pl-10" placeholder="Search issues" aria-label="Search issues" value={query} onChange={e => setQuery(e.target.value)} /></label><p className="eyebrow mt-4">{filtered.length} canonical issues</p></div><div className="overflow-y-auto">{filtered.length ? filtered.map((issue, index) => <button key={issue.canonical_id} onClick={() => onSelect(issue.canonical_id)} className={`grid w-full grid-cols-[24px_minmax(0,1fr)_auto] items-center gap-3 border-b border-line px-5 py-4 text-left text-sm hover:bg-sand/60 ${active?.canonical_id === issue.canonical_id ? 'bg-sand' : ''}`}><span className="font-mono text-[10px] text-orange">{String(index + 1).padStart(2, '0')}</span><span className="truncate font-semibold">{issue.canonical_name ?? 'Unnamed issue'}</span><span className="font-mono text-xs text-muted tabular">{integer(issue.review_count)}</span></button>) : <p className="p-5 text-sm text-muted">No matching issues</p>}</div></aside>
      <div className="panel min-w-0 p-6 md:p-9">{active ? <IssueDetail issue={active} entry={entry} onReviewClick={onReviewClick} /> : <div className="flex h-full items-center justify-center text-sm text-muted">No issues identified in this sample.</div>}</div>
    </div>
    <section className="mt-16"><div className="mb-5 flex items-baseline gap-4 border-b border-ink pb-4"><span className="font-mono text-xs text-orange">02</span><h3 className="section-heading">Issue metrics</h3></div><div className="panel overflow-x-auto px-5"><table className="data-table min-w-[850px]"><thead><tr>{([['canonical_name','Issue'],['review_count','Mentions'],['share_of_successful_reviews','Share'],['average_rating','Average rating'],['negative_rating_share','1–2★ share'],['category','Category']] as [SortKey,string][]).map(([key, title]) => <th key={key}><button className="hover:text-ink" onClick={() => setSort(s => ({ key, direction: s.key === key ? (s.direction === 1 ? -1 : 1) : -1 }))}>{title} {sort.key === key ? (sort.direction === 1 ? '↑' : '↓') : ''}</button></th>)}</tr></thead><tbody>{sorted.map(issue => <tr key={issue.canonical_id} className="cursor-pointer" onClick={() => { onSelect(issue.canonical_id); window.scrollTo({ top: 280, behavior: 'smooth' }) }}><td className="font-semibold">{issue.canonical_name ?? 'Unnamed issue'}</td><td className="font-mono tabular">{integer(issue.review_count)}</td><td className="font-mono tabular">{percent(issue.share_of_successful_reviews)}</td><td className="font-mono tabular">{decimal(issue.average_rating)}</td><td className="font-mono tabular">{percent(issue.negative_rating_share)}</td><td>{label(issue.category)}</td></tr>)}</tbody></table>{!sorted.length && <p className="py-10 text-sm text-muted">Not enough data</p>}</div></section>
    <FeatureRequests features={features} catalog={catalog?.feature_requests ?? []} onReviewClick={onReviewClick} />
  </div>
}

function IssueDetail({ issue, entry, onReviewClick }: { issue: IssueMetric; entry?: CatalogEntry; onReviewClick: (ids: string[]) => void }) {
  const formulations = useMemo(() => Array.from(new Set((entry?.source_issues ?? []).flatMap(source => [...(source.descriptions ?? []), ...(source.description ? [source.description] : [])]).filter(Boolean))), [entry])
  const evidence = useMemo(() => Array.from(new Set((entry?.source_issues ?? []).flatMap(source => source.evidence ?? []).filter(Boolean))), [entry])
  const ids = issue.review_ids ?? entry?.review_ids ?? []
  return <><p className="eyebrow">Issue / {label(issue.category)}</p><h3 className="mt-3 max-w-[740px] font-serif text-[clamp(32px,3.2vw,48px)] leading-tight">{issue.canonical_name ?? 'Unnamed issue'}</h3><div className="mt-9 grid grid-cols-2 gap-y-6 border-y border-line py-6 md:grid-cols-4"><Metric value={integer(issue.review_count)} name="Reviews" /><Metric value={percent(issue.share_of_successful_reviews)} name="Of analyzed sample" /><Metric value={decimal(issue.average_rating)} name="Avg rating" /><Metric value={percent(issue.negative_rating_share)} name="1–2★ share" /></div>
    <div className="mt-9 grid gap-8 md:grid-cols-2"><div><p className="eyebrow">Category</p><p className="mt-2 text-sm font-semibold">{label(issue.category)}</p></div><div><p className="eyebrow">Rating sample</p><p className="mt-2 text-sm font-semibold">{integer(issue.rating_sample_size)} reviews with a rating</p></div></div>
    {formulations.length > 0 && <div className="mt-10 border-t border-line pt-6"><p className="eyebrow">Source formulations</p><ul className="mt-4 space-y-3">{formulations.map(text => <li key={text} className="border-l-2 border-ochre pl-4 text-sm leading-relaxed">{text}</li>)}</ul></div>}
    {evidence.length > 0 && <div className="mt-9 border-t border-line pt-6"><p className="eyebrow">Review evidence</p><ul className="mt-4 grid gap-3 md:grid-cols-2">{evidence.slice(0, 6).map((text, i) => <li key={`${text}-${i}`} className="bg-paper p-4 text-sm italic leading-relaxed text-muted">“{text}”</li>)}</ul></div>}
    <div className="mt-9 border-t border-line pt-6"><p className="eyebrow">Related reviews</p><p className="mt-2 text-sm text-muted">{integer(ids.length)} linked review IDs in the issue catalog.</p>{ids.length > 0 && <button className="thin-button mt-4" onClick={() => onReviewClick(ids)}>View supporting reviews <ArrowRight size={15} /></button>}</div>
  </>
}

function Metric({ value, name }: { value: string; name: string }) { return <div><strong className="block font-mono text-2xl font-medium tabular md:text-3xl">{value}</strong><span className="eyebrow mt-2 block">{name}</span></div> }

function FeatureRequests({ features, catalog, onReviewClick }: { features: FeatureRequestMetric[]; catalog: CatalogEntry[]; onReviewClick: (ids: string[]) => void }) {
  const [sort, setSort] = useState<'review_count' | 'share_of_successful_reviews'>('review_count')
  const sorted = [...features].sort((a,b) => (b[sort] ?? -1) - (a[sort] ?? -1))
  return <section className="mt-16"><div className="mb-5 flex items-baseline gap-4 border-b border-ink pb-4"><span className="font-mono text-xs text-orange">03</span><h3 className="section-heading">Feature requests</h3></div><div className="panel overflow-x-auto px-5"><table className="data-table min-w-[700px]"><thead><tr><th>Request</th><th><button onClick={() => setSort('review_count')}>Mentions {sort === 'review_count' ? '↓' : ''}</button></th><th><button onClick={() => setSort('share_of_successful_reviews')}>Share {sort === 'share_of_successful_reviews' ? '↓' : ''}</button></th><th>Category</th><th>Evidence</th></tr></thead><tbody>{sorted.map(item => { const entry = catalog.find(x => x.canonical_id === item.canonical_id); const ids = item.review_ids ?? entry?.review_ids ?? []; return <tr key={item.canonical_id}><td className="font-semibold">{item.canonical_name ?? 'Unnamed request'}</td><td className="font-mono tabular">{integer(item.review_count)}</td><td className="font-mono tabular">{percent(item.share_of_successful_reviews)}</td><td>{label(item.category)}</td><td>{ids.length ? <button className="text-rust underline underline-offset-2" onClick={() => onReviewClick(ids)}>View reviews</button> : '—'}</td></tr> })}</tbody></table>{!sorted.length && <p className="py-10 text-sm text-muted">No feature requests identified in this sample.</p>}</div></section>
}
