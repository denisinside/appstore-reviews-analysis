import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { getInsights, getIssues, getMetrics, getScan } from '../api/client'
import type { MetricsResponse, Scan } from '../api/types'
import { decimal, integer, label, languageName, percent } from '../utils/format'
import { AspectSentiment, AverageRatingByCountry, FeatureRequests, IssueImpactScatter, LanguageDistribution, RatingByVersion, RatingDistribution, ReviewsByCountry, SentimentDistribution, TopIssues } from './charts'
import '../report.css'

function Panel({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return <div className={`report-panel ${className}`}><h3>{title}</h3>{children}</div>
}

function Chart({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return <Panel title={title} className={`report-chart ${className}`}><div className="report-chart-canvas">{children}</div></Panel>
}

function Section({ number, title, children, className = '' }: { number: string; title: string; children: React.ReactNode; className?: string }) {
  return <section className={`report-section ${className}`}><div className="report-section-heading"><span>{number}</span><h2>{title}</h2></div>{children}</section>
}

function useReportData(scanId: string) {
  const scan = useQuery({ queryKey: ['scan', scanId], queryFn: () => getScan(scanId), enabled: !!scanId })
  const complete = scan.data?.analysis_status === 'completed'
  const metrics = useQuery({ queryKey: ['metrics', scanId], queryFn: () => getMetrics(scanId), enabled: complete })
  const insights = useQuery({ queryKey: ['insights', scanId], queryFn: () => getInsights(scanId), enabled: complete })
  const issues = useQuery({ queryKey: ['issues', scanId], queryFn: () => getIssues(scanId), enabled: complete })
  const ready = !!scan.data && !!metrics.data && !!insights.data && !!issues.data && complete && !metrics.isLoading && !insights.isLoading && !issues.isLoading
  return { scan, metrics, insights, issues, ready }
}

export default function ReportPage({ scanId: suppliedScanId }: { scanId?: string }) {
  const params = useParams()
  const scanId = suppliedScanId ?? params.scanId ?? ''
  const { scan, metrics, insights, issues, ready } = useReportData(scanId)
  if (scan.isLoading || (scan.data?.analysis_status === 'completed' && (metrics.isLoading || insights.isLoading || issues.isLoading))) return <main className="report-loading">Preparing report…</main>
  if (scan.error || metrics.error || insights.error || issues.error || !scan.data) return <main className="report-loading" role="alert">Report data could not be loaded.</main>
  if (!ready || !metrics.data || !insights.data || !issues.data) return <main className="report-loading">This scan does not have a completed analysis.</main>
  return <Report scan={scan.data} metrics={metrics.data} insights={insights.data} issues={issues.data} ready={ready} />
}

function Report({ scan, metrics, insights, issues, ready }: { scan: Scan; metrics: MetricsResponse; insights: Awaited<ReturnType<typeof getInsights>>; issues: Awaited<ReturnType<typeof getIssues>>; ready: boolean }) {
  const reportRef = useRef<HTMLElement>(null)
  const [chartsRendered, setChartsRendered] = useState(false)
  useEffect(() => {
    if (!ready) return
    let secondFrame = 0
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => {
        const root = reportRef.current
        if (root && root.querySelector('.recharts-surface')) setChartsRendered(true)
      })
    })
    return () => { cancelAnimationFrame(firstFrame); cancelAnimationFrame(secondFrame) }
  }, [ready])
  const { basic, sentiment, nlp } = metrics
  const issueRows = [...(nlp?.issue_metrics.issues ?? [])].sort((a, b) => b.review_count - a.review_count)
  const featureRows = [...(nlp?.feature_request_metrics.feature_requests ?? [])].sort((a, b) => b.review_count - a.review_count)
  const scope = scan.collection_mode === 'country' ? `${scan.country?.toUpperCase() ?? 'Country'} storefront` : `Top ${scan.top_n ?? '—'} countries`
  return <main ref={reportRef} className="analysis-report" data-report-ready={chartsRendered ? 'true' : undefined}>
    <header className="report-cover"><div className="report-brand"><span>f.</span> fieldnotes <i>/</i> App Store intelligence</div><p className="report-kicker">Review analysis report</p><h1>{scan.app_name || `App ${scan.app_id}`}<em>Review analysis</em></h1><p className="report-deck">A structured read of what customers praise, what frustrates them, and where product teams can respond.</p><div className="report-cover-meta"><div><small>App Store ID</small><strong>{scan.app_id}</strong></div><div><small>Scan ID</small><strong>{scan.scan_id}</strong></div><div><small>Created</small><strong>{new Date(scan.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</strong></div><div><small>Collection scope</small><strong>{scope}</strong></div></div><p className="report-sample">{integer(scan.review_count)} collected reviews · {scan.max_pages} pages per storefront</p></header>
    <Section number="01" title="Overall summary" className="report-summary"><p className="report-lede">{insights.overall_summary || 'No overall summary was generated for this sample.'}</p><p className="report-note">This report combines collected review ratings, text sentiment, normalized issues, and generated recommendations.</p></Section>
    <Section number="02" title="Key metrics"><div className="report-kpis">{[
      ['Reviews', integer(basic.review_count)], ['Average rating', `${decimal(basic.average_rating)} ★`], ['1–2★ reviews', percent(basic.negative_rating_share)], ['Canonical issues', integer(nlp?.issue_metrics.issues.length ?? null)], ['Sentiment sample', integer(sentiment.sample_size)], ['Analysis coverage', `${integer(nlp?.analysis_summary.analyzed_review_count)} / ${integer(nlp?.analysis_summary.input_review_count)}`],
    ].map(([name, value]) => <div key={name}><strong>{value}</strong><span>{name}</span></div>)}</div></Section>
    <Section number="03" title="Rating & sentiment"><div className="report-chart-grid"><Chart title="Rating distribution"><RatingDistribution data={basic.rating_distribution} /></Chart><Chart title="Sentiment distribution"><SentimentDistribution distribution={sentiment.distribution} reviewCount={sentiment.sample_size} staticChart /></Chart></div></Section>
    <Section number="04" title="Aspect sentiment"><Chart title="Positive, neutral, and negative mentions"><AspectSentiment data={nlp?.aspect_metrics} /></Chart></Section>
    <Section number="05" title="Markets, languages & versions"><div className="report-chart-grid"><Chart title="Reviews by country"><ReviewsByCountry data={basic.country_statistics} /></Chart><Chart title="Average rating by country"><AverageRatingByCountry data={basic.country_statistics} /></Chart><Chart title="Language distribution"><LanguageDistribution data={basic.language_statistics} /></Chart><Chart title="Rating by app version"><RatingByVersion data={basic.rating_by_version} /></Chart></div><div className="report-chart-grid report-tables"><Table title="Country details" headers={['Country', 'Reviews', 'Avg rating']} rows={basic.country_statistics.slice(0, 14).map(row => [row.country?.toUpperCase() || 'Unknown', integer(row.review_count), decimal(row.average_rating)])} /><Table title="Language details" headers={['Language', 'Reviews', 'Avg rating']} rows={basic.language_statistics.slice(0, 14).map(row => [languageName(row.language), integer(row.review_count), decimal(row.average_rating)])} /></div></Section>
    <Section number="06" title="Top issues"><Chart title="Most frequent issues" className="report-top-issues-chart"><TopIssues data={issueRows} limit={8} expandedLabels /></Chart></Section>
    <Section number="07" title="Issue impact"><Chart title="Frequency and average rating" className="report-impact-chart"><IssueImpactScatter data={issueRows} numbered /></Chart><p className="report-note">Each numbered point shows issue frequency and average rating; issues with identical values share a point. Bubble size reflects review count. Small groups should be interpreted with care.</p></Section>
    <Section number="08" title="Feature requests"><div className="report-chart-grid"><Chart title="Most requested features"><FeatureRequests data={featureRows} limit={8} /></Chart><Table title="Request metrics" headers={['Feature request', 'Reviews', 'Share', 'Category']} rows={featureRows.slice(0, 12).map(row => [row.canonical_name || 'Unnamed request', integer(row.review_count), percent(row.share_of_successful_reviews), label(row.category)])} /></div></Section>
    <Section number="09" title="Issue metrics"><Table title="Issue frequency and rating impact" headers={['Issue', 'Reviews', 'Share', 'Avg rating', '1–2★ share', 'Category']} rows={issueRows.slice(0, 25).map(row => [row.canonical_name || 'Unnamed issue', integer(row.review_count), percent(row.share_of_successful_reviews), decimal(row.average_rating), percent(row.negative_rating_share), label(row.category)])} /></Section>
    <Section number="10" title="Actionable insights"><p className="report-note">Priorities are based on recurring themes and representative reviews in this sample.</p>{insights.issue_insights.slice(0, 8).map((insight, index) => { const entry = issues.issues.find(item => item.canonical_id === insight.canonical_id); const evidence = (entry?.source_issues ?? []).flatMap(item => item.evidence ?? []).slice(0, 2); return <article className="report-insight" key={insight.canonical_id}><div className="report-insight-number">{String(index + 1).padStart(2, '0')}</div><div><p className="report-eyebrow">{label(insight.category)} · {integer(insight.metrics.review_count)} reviews</p><h3>{insight.canonical_name || 'Priority issue'}</h3><p>{insight.finding}</p><p><strong>User impact: </strong>{insight.user_impact}</p><div className="report-actions"><strong>Recommended actions</strong><ul>{insight.recommended_actions.slice(0, 4).map((action, i) => <li key={i}>{action}</li>)}</ul></div>{evidence.map((text, i) => <blockquote key={i}>“{text}”</blockquote>)}</div></article> })}{insights.feature_request_insights.slice(0, 4).map(insight => <article className="report-insight report-feature-insight" key={insight.canonical_id}><div className="report-insight-number">↗</div><div><p className="report-eyebrow">Product opportunity · {integer(insight.metrics.review_count)} reviews</p><h3>{insight.canonical_name || 'Feature request'}</h3><p>{insight.finding}</p><p><strong>Recommended action: </strong>{insight.recommended_action}</p></div></article>)}</Section>
    <Section number="11" title="Methodology & limitations" className="report-method"><p>Ratings are sourced from the collected App Store review sample. Sentiment, aspects, issues, and feature requests come from the completed text analysis and are limited to reviews processed successfully.</p><p>Issue and request shares use successfully analyzed reviews as the denominator. Reviews may mention multiple aspects or issues, so category counts can overlap. Country counts can overlap when one review was observed in multiple storefronts.</p><ul>{insights.limitations.map((item, index) => <li key={index}>{item}</li>)}</ul>{nlp?.analysis_summary.rules?.length ? <><h3>Analysis rules</h3><ul>{nlp.analysis_summary.rules.slice(0, 8).map((item, index) => <li key={index}>{item}</li>)}</ul></> : null}<p className="report-generated">Generated from scan {scan.scan_id} · Fieldnotes App Store intelligence</p></Section>
  </main>
}

function Table({ title, headers, rows }: { title: string; headers: string[]; rows: string[][] }) {
  return <Panel title={title} className="report-table-wrap"><table className="report-table"><thead><tr>{headers.map(header => <th key={header}>{header}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, index) => <tr key={index}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>) : <tr><td colSpan={headers.length}>Not enough data</td></tr>}</tbody></table></Panel>
}
