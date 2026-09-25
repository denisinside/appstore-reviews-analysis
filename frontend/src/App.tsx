import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowDownToLine, ArrowLeft, ArrowRight, CircleAlert, LoaderCircle, MoveUpRight } from 'lucide-react'
import { Link, Route, Routes, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { analyzeScan, createScan, downloadReviewsUrl, getInsights, getIssues, getMetrics, getScan } from './api/client'
import type { Scan } from './api/types'
import { parseAppStoreId } from './utils/appId'
import { integer } from './utils/format'
import Overview from './components/Overview'
import IssuesView from './components/IssuesView'
import InsightsView from './components/InsightsView'
import ReviewsExplorer from './components/ReviewsExplorer'

const tabs = ['overview', 'issues', 'insights', 'reviews'] as const
type Tab = typeof tabs[number]

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="min-h-screen bg-paper"><header className="border-b border-line bg-surface"><div className="mx-auto flex max-w-[1480px] items-center justify-between px-5 py-4 md:px-10">
    <Link to="/" className="flex items-center gap-3 text-ink no-underline"><span className="flex h-8 w-8 items-center justify-center bg-orange font-serif text-lg text-white">f.</span><span className="text-sm font-bold tracking-[-0.02em]">fieldnotes<span className="text-orange">/</span></span><span className="hidden border-l border-line pl-3 font-mono text-[10px] uppercase tracking-[0.14em] text-muted sm:block">App Store intelligence</span></Link>
    <span className="eyebrow hidden sm:block">Review analysis studio · v1.0</span>
  </div></header>{children}</div>
}

function Home() {
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<'top' | 'country'>('top')
  const [country, setCountry] = useState('US')
  const [topN, setTopN] = useState(10)
  const [maxPages, setMaxPages] = useState(10)
  const appId = parseAppStoreId(input)
  const mutation = useMutation({ mutationFn: createScan, onSuccess: scan => navigate(`/scans/${scan.scan_id}?start=1`) })
  const submit = (event: React.FormEvent) => { event.preventDefault(); if (!appId) return; mutation.mutate({ app_id: appId, mode, country: mode === 'country' ? country.trim().toLowerCase() : undefined, top_n: mode === 'top' ? topN : undefined, max_pages: maxPages }) }

  return <Shell><main className="mx-auto max-w-[1480px] px-5 pb-20 pt-12 md:px-10 md:pt-20">
    <div className="grid gap-12 lg:grid-cols-[minmax(0,0.88fr)_minmax(480px,1.12fr)] lg:gap-20">
      <div className="max-w-[580px]"><p className="eyebrow mb-5 flex items-center gap-3"><span className="h-[7px] w-[7px] bg-orange" />New analysis / 001</p><h1 className="font-serif text-[clamp(48px,6vw,88px)] leading-[0.99] tracking-[-0.045em]">Read between<br />the <em className="font-medium text-orange">ratings.</em></h1><p className="mt-7 max-w-[470px] text-lg leading-relaxed text-muted">A focused workspace for understanding what people praise, what frustrates them, and which product issues deserve attention.</p>
        <div className="mt-12 grid grid-cols-3 gap-5 border-y border-line py-6 text-sm"><div><span className="eyebrow block">01 / Collect</span><span className="mt-2 block">Public App Store reviews</span></div><div><span className="eyebrow block">02 / Analyze</span><span className="mt-2 block">Language & themes</span></div><div><span className="eyebrow block">03 / Decide</span><span className="mt-2 block">Evidence-backed actions</span></div></div>
      </div>
      <form onSubmit={submit} className="panel self-start p-6 md:p-9"><div className="flex items-start justify-between gap-4 border-b border-line pb-6"><div><p className="eyebrow">Set up your scan</p><h2 className="mt-2 font-serif text-3xl">Analyze App Store reviews</h2></div><span className="font-mono text-xs text-orange">↗ 01</span></div>
        <div className="mt-7"><label htmlFor="app-id" className="mb-2 block text-sm font-semibold">App Store ID or link</label><input id="app-id" className="field h-14 font-mono text-[15px]" placeholder="512939461 or apps.apple.com/…" value={input} onChange={e => setInput(e.target.value)} aria-invalid={input.length > 0 && !appId} />
          <div className="mt-2 min-h-5 font-mono text-xs">{appId ? <span className="text-good">✓ App ID: {appId}</span> : input ? <span className="text-danger">Enter a numeric ID or Apple App Store link.</span> : <span className="text-muted">Paste an App Store URL or enter its numeric ID.</span>}</div></div>
        <fieldset className="mt-7"><legend className="mb-3 text-sm font-semibold">Scan scope</legend><div className="grid grid-cols-2 border border-line"><button type="button" onClick={() => setMode('top')} className={`px-4 py-3 text-sm font-semibold ${mode === 'top' ? 'bg-ink text-surface' : 'bg-surface text-muted hover:text-ink'}`}>Top countries</button><button type="button" onClick={() => setMode('country')} className={`border-l border-line px-4 py-3 text-sm font-semibold ${mode === 'country' ? 'bg-ink text-surface' : 'bg-surface text-muted hover:text-ink'}`}>Single country</button></div></fieldset>
        <div className="mt-6 grid gap-5 sm:grid-cols-2"><label className="block text-sm font-semibold">{mode === 'top' ? 'Top countries' : 'Country'}{mode === 'top' ? <input className="field mt-2" type="number" min="1" max="32" value={topN} onChange={e => setTopN(Number(e.target.value))} /> : <input className="field mt-2 uppercase" maxLength={2} value={country} onChange={e => setCountry(e.target.value.toUpperCase())} placeholder="US" />}</label><label className="block text-sm font-semibold">Pages per country<input className="field mt-2" type="number" min="1" max="10" value={maxPages} onChange={e => setMaxPages(Number(e.target.value))} /></label></div>
        <div className="mt-8 border-t border-line pt-6"><button className="accent-button w-full justify-between" disabled={!appId || mutation.isPending || (mode === 'country' && !/^[A-Za-z]{2}$/.test(country)) || topN < 1 || topN > 32 || maxPages < 1 || maxPages > 10} type="submit"><span>{mutation.isPending ? 'Collecting reviews…' : 'Run analysis'}</span>{mutation.isPending ? <LoaderCircle size={18} className="animate-spin" /> : <ArrowRight size={18} />}</button><p className="mt-3 text-xs leading-relaxed text-muted">{mutation.isPending ? 'Collection happens first and can take a few minutes for multiple countries. Keep this page open.' : 'Analysis starts after the review sample is collected.'}</p>{mutation.isError && <p role="alert" className="mt-4 border-l-2 border-danger bg-danger/5 p-3 text-sm text-danger">{(mutation.error as Error).message}</p>}</div>
      </form>
    </div>
    <div className="mt-20 grid gap-7 border-t border-line pt-8 text-sm text-muted md:grid-cols-[1fr_2fr]"><span className="eyebrow">A clearer reading of feedback</span><p className="max-w-2xl leading-relaxed">Ratings tell you how people feel. The review text explains why. Fieldnotes connects sentiment, recurring issues, feature requests, and original evidence in one place.</p></div>
  </main></Shell>
}

function StatusPanel({ scan, onRetry, retrying, retryError }: { scan: Scan; onRetry: () => void; retrying: boolean; retryError?: string }) {
  const failed = scan.analysis_status === 'failed'
  return <div className="mx-auto mt-14 max-w-[740px] panel"><div className="p-8 md:p-12"><p className={`eyebrow ${failed ? '!text-danger' : '!text-orange'}`}>{failed ? 'Analysis failed' : scan.analysis_status === 'not_started' ? 'Analysis ready' : 'Analysis running'}</p><h2 className="mt-4 font-serif text-4xl">{failed ? 'The analysis stopped.' : scan.analysis_status === 'not_started' ? 'Ready to analyze.' : 'Reading the review sample.'}</h2><div className="mt-9 grid grid-cols-2 gap-6 border-y border-line py-6"><div><span className="eyebrow">Reviews collected</span><strong className="mt-2 block font-mono text-4xl font-medium tabular">{integer(scan.review_count)}</strong></div><div><span className="eyebrow">Collection</span><strong className="mt-3 block text-sm font-semibold capitalize">{scan.collection_status ?? 'Unknown'}</strong></div></div><p className="mt-7 leading-relaxed text-muted">{failed ? scan.error || 'The backend did not provide an error message.' : scan.analysis_status === 'not_started' ? 'The review sample is collected. Start analysis to process sentiment, keywords, issues and actionable insights.' : 'Processing sentiment, keywords, issues and actionable insights. This page updates automatically.'}</p>{(failed || scan.analysis_status === 'not_started') && <button className="thin-button mt-6" onClick={onRetry} disabled={retrying}>{failed ? 'Retry analysis' : 'Start analysis'} <ArrowRight size={15} /></button>}{retryError && <p role="alert" className="mt-3 text-sm text-danger">{retryError}</p>}</div>{(scan.analysis_status === 'queued' || scan.analysis_status === 'running') && <div className="activity-line" />}</div>
}

function Workspace() {
  const { scanId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const queryClient = useQueryClient()
  const requested = useRef(false)
  const status = useQuery({ queryKey: ['scan', scanId], queryFn: () => getScan(scanId), enabled: !!scanId, refetchInterval: query => ['queued', 'running'].includes(query.state.data?.analysis_status ?? '') ? 2_000 : false })
  const scan = status.data
  const analysis = useMutation({ mutationFn: () => analyzeScan(scanId), onSuccess: () => { queryClient.setQueryData<Scan>(['scan', scanId], old => old ? { ...old, analysis_status: 'queued' } : old); void status.refetch() } })
  useEffect(() => { if (params.get('start') === '1' && scan?.analysis_status === 'not_started' && !requested.current) { requested.current = true; analysis.mutate() } }, [scan?.analysis_status, params, analysis])
  const completed = scan?.analysis_status === 'completed'
  const metrics = useQuery({ queryKey: ['metrics', scanId], queryFn: () => getMetrics(scanId), enabled: completed })
  const insights = useQuery({ queryKey: ['insights', scanId], queryFn: () => getInsights(scanId), enabled: completed })
  const issues = useQuery({ queryKey: ['issues', scanId], queryFn: () => getIssues(scanId), enabled: completed })
  const current = tabs.includes(params.get('tab') as Tab) ? params.get('tab') as Tab : 'overview'
  const navigateTab = (tab: Tab, extra?: Record<string, string>) => setParams({ tab, ...(extra ?? {}) })
  const reviewIds = params.get('review')?.split(',').filter(Boolean)
  const scope = scan?.collection_mode === 'country' ? `${scan.country?.toUpperCase() ?? 'Country'} storefront` : `Top ${scan?.top_n ?? '—'} countries`

  return <Shell><main className="mx-auto max-w-[1480px] px-5 pb-20 pt-8 md:px-10"><Link to="/" className="inline-flex items-center gap-2 text-sm text-muted hover:text-rust"><ArrowLeft size={15} /> New analysis</Link>
    {status.isLoading ? <div className="mt-20 flex items-center gap-3 text-muted"><LoaderCircle size={18} className="animate-spin" /> Loading scan…</div> : status.isError || !scan ? <div className="panel mt-12 p-8"><CircleAlert className="text-danger" /><h2 className="mt-4 font-serif text-3xl">Scan unavailable</h2><p className="mt-2 text-muted">{(status.error as Error)?.message ?? 'The scan could not be found.'}</p></div> : <>
      <div className="mt-9 flex flex-wrap items-end justify-between gap-6 border-b border-ink pb-8"><div><p className="eyebrow">APP {scan.app_id} / SCAN {scan.scan_id}</p><h1 className="mt-3 font-serif text-[clamp(38px,4.6vw,64px)] leading-none tracking-[-0.035em]">App Store Review Analysis</h1><p className="mt-5 text-sm text-muted"><span className="font-mono text-ink">{integer(scan.review_count)}</span> reviews <span className="mx-2 text-line">·</span> {scope}</p></div><div className="flex flex-wrap items-center gap-5"><span className={`font-mono text-[11px] font-medium uppercase tracking-[0.14em] ${completed ? 'text-good' : scan.analysis_status === 'failed' ? 'text-danger' : 'text-orange'}`}><span className="mr-2 inline-block h-2 w-2 rounded-full bg-current" />{scan.analysis_status.replaceAll('_', ' ')}</span><a href={downloadReviewsUrl(scanId)} className="thin-button"><ArrowDownToLine size={16} /> Download reviews</a></div></div>
      {!completed ? <StatusPanel scan={scan} onRetry={() => { requested.current = true; analysis.mutate() }} retrying={analysis.isPending} retryError={analysis.isError ? (analysis.error as Error).message : undefined} /> : <>
        <nav aria-label="Analysis sections" className="mt-1 flex gap-7 overflow-x-auto border-b border-line">{tabs.map(tab => <button key={tab} className={`whitespace-nowrap border-b-2 px-1 py-5 text-sm font-semibold capitalize ${current === tab ? 'border-orange text-ink' : 'border-transparent text-muted hover:text-ink'}`} onClick={() => navigateTab(tab)}>{tab}</button>)}</nav>
        <div className="pt-10">{current === 'overview' && <Overview metrics={metrics.data} loading={metrics.isLoading} error={metrics.error as Error | null} onIssueClick={id => navigateTab('issues', { issue: id })} />}{current === 'issues' && <IssuesView metrics={metrics.data} catalog={issues.data} loading={metrics.isLoading || issues.isLoading} error={(metrics.error || issues.error) as Error | null} selectedId={params.get('issue') ?? undefined} onSelect={id => navigateTab('issues', { issue: id })} onReviewClick={ids => navigateTab('reviews', { review: ids.join(',') })} />}{current === 'insights' && <InsightsView data={insights.data} loading={insights.isLoading} error={insights.error as Error | null} onReviewClick={ids => navigateTab('reviews', { review: ids.join(',') })} />}{current === 'reviews' && <ReviewsExplorer scanId={scanId} focusReviewIds={reviewIds} onCloseFocus={() => navigateTab('reviews')} />}</div>
      </>}
      <div className="mt-20 flex items-center justify-between border-t border-line pt-5 text-xs text-muted"><span>Based on the collected App Store review sample.</span><span className="inline-flex items-center gap-1">Fieldnotes <MoveUpRight size={12} /></span></div>
    </>}
  </main></Shell>
}

export default function App() { return <Routes><Route path="/" element={<Home />} /><Route path="/scans/:scanId" element={<Workspace />} /><Route path="*" element={<Shell><main className="mx-auto max-w-4xl px-6 py-24"><p className="eyebrow">404 / Page not found</p><h1 className="mt-3 font-serif text-5xl">This page isn’t here.</h1><Link className="accent-button mt-8" to="/">Start a new analysis <ArrowRight size={16} /></Link></main></Shell>} /></Routes> }
