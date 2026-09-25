import { useQuery } from '@tanstack/react-query'
import { ArrowUpRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { getScans } from '../api/client'
import type { ScanSummary } from '../api/types'
import { decimal, integer, percent } from '../utils/format'

const dateLabel = (value: string | null) => {
  if (!value) return 'Date unknown'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function ScanRow({ scan }: { scan: ScanSummary }) {
  const scope = scan.collection_mode === 'country'
    ? `${scan.country?.toUpperCase() ?? 'Country'} storefront`
    : `Top ${scan.top_n ?? '—'} countries`
  const statusColor = scan.analysis_status === 'completed' ? 'text-good' : scan.analysis_status === 'failed' ? 'text-danger' : 'text-orange'
  return <Link to={`/scans/${encodeURIComponent(scan.scan_id)}`} className="group block border-t border-line py-6 text-ink transition-colors hover:bg-sand/40 hover:px-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-orange md:grid md:grid-cols-[minmax(180px,1.1fr)_minmax(0,2fr)_auto] md:items-center md:gap-6">
    <div><h3 className="font-serif text-2xl leading-tight group-hover:text-rust">{scan.app_name || `App ${scan.app_id}`}</h3><p className="mt-1 font-mono text-xs text-muted">ID {scan.app_id}</p></div>
    <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 md:mt-0">
      <span><strong className="font-mono text-lg font-medium tabular">{integer(scan.review_count)}</strong><small className="ml-2 text-xs text-muted">reviews</small></span>
      {scan.average_rating != null && <span><strong className="font-mono text-lg font-medium tabular">{decimal(scan.average_rating)} ★</strong><small className="ml-2 text-xs text-muted">avg</small></span>}
      {scan.negative_sentiment_share != null && <span><strong className="font-mono text-lg font-medium tabular">{percent(scan.negative_sentiment_share, 0)}</strong><small className="ml-2 text-xs text-muted">negative sentiment</small></span>}
      {scan.issue_count != null && <span><strong className="font-mono text-lg font-medium tabular">{integer(scan.issue_count)}</strong><small className="ml-2 text-xs text-muted">issues</small></span>}
    </div>
    <div className="mt-4 flex items-center justify-between gap-5 md:mt-0 md:justify-end"><div className="text-left md:text-right"><p className="text-xs text-muted">{scope}</p><p className="mt-1 font-mono text-[11px] text-muted">{dateLabel(scan.created_at)}</p><p className={`mt-2 font-mono text-[11px] uppercase tracking-[0.12em] ${statusColor}`}>{scan.analysis_status.replaceAll('_', ' ')}</p></div><ArrowUpRight size={17} className="text-orange" /></div>
  </Link>
}

export default function RecentAnalyses() {
  const scans = useQuery({ queryKey: ['scans'], queryFn: getScans, refetchInterval: query => query.state.data?.items.some(scan => scan.analysis_status === 'queued' || scan.analysis_status === 'running') ? 5_000 : false })
  const items = [...(scans.data?.items ?? [])].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
  return <section className="mt-20"><div className="mb-6 flex items-end justify-between gap-4 border-b border-ink pb-4"><div><p className="eyebrow">Your workspace</p><h2 className="section-heading mt-2">Recent analyses</h2></div><span className="font-mono text-xs text-muted">{scans.isSuccess ? `${items.length} saved` : 'Scan history'}</span></div>
    {scans.isLoading ? <p className="py-8 text-sm text-muted">Loading analyses…</p> : scans.isError ? <p role="alert" className="py-8 text-sm text-danger">Could not load analyses: {(scans.error as Error).message}</p> : items.length ? <div className="border-b border-line">{items.map(scan => <ScanRow key={scan.scan_id} scan={scan} />)}</div> : <p className="py-12 text-sm text-muted">No analyses yet</p>}
  </section>
}
