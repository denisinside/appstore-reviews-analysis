import { useMemo } from 'react';
import type { ReactNode } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, Pie, PieChart,
  ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import {
  aspectChartRows, CHART_COLORS, DistributionRow, issueImpactRows, issuesOverTimeRows,
  MetricIssue, ratingChartRows, RatingBucket, SENTIMENT_COLORS, sentimentChartRows,
} from './chartData';
import type { MetricsResponse } from '../../api/types';

const axis = { fontSize: 11, fill: '#756a5f' };
const grid = '#e8dfd4';
const tooltipStyle = { background: '#fffcf6', border: '1px solid #ddd1c3', borderRadius: 4, color: '#211c17', fontSize: 12 };
const compact = (n: number) => new Intl.NumberFormat().format(n);
const shortLabel = (text: string, max = 22) => text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
const pct = (n: number | null | undefined) => n == null ? 'No data' : `${(n * 100).toFixed(1)}%`;
const rating = (n: number | null | undefined) => n == null ? 'No data' : `${n.toFixed(2)} ★`;

export function ChartFrame({ children, className = '' }: { title?: string; subtitle?: string; children: ReactNode; className?: string }) {
  return <div className={`min-w-0 ${className}`}>{children}</div>;
}
function Empty({ children = 'Not enough data' }: { children?: string }) { return <div className="flex h-56 items-center justify-center border border-dashed border-[#ddd1c3] text-sm text-[#756a5f]">{children}</div>; }

export function RatingDistribution({ data = [] }: { data?: RatingBucket[] }) {
  const rows = ratingChartRows(data); const total = rows.reduce((n, row) => n + row.count, 0);
  return <ChartFrame title="Rating distribution" subtitle="Collected review ratings"><div className="h-56">{total ? <ResponsiveContainer><BarChart data={rows} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
    <CartesianGrid stroke={grid} vertical={false} /><XAxis dataKey="rating" tick={axis} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tick={axis} axisLine={false} tickLine={false} />
    <Tooltip contentStyle={tooltipStyle} formatter={(value, _name, item) => [`${compact(Number(value))} reviews${item.payload.share == null ? '' : ` · ${pct(item.payload.share)}`}`, 'Count']} />
    <Bar dataKey="count" fill="#e86f21" maxBarSize={44} radius={[2, 2, 0, 0]} />
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}

export function SentimentDistribution({ distribution, reviewCount }: { distribution?: Partial<Record<'positive' | 'neutral' | 'negative', { count: number; share: number | null }>>; reviewCount?: number }) {
  const rows = sentimentChartRows(distribution); const total = rows.reduce((n, row) => n + row.count, 0);
  return <ChartFrame title="Sentiment" subtitle="Overall sentiment classification"><div className="relative h-56">{total ? <><ResponsiveContainer><PieChart>
    <Pie data={rows} dataKey="count" nameKey="name" innerRadius={60} outerRadius={86} paddingAngle={2} stroke="none">{rows.map((r) => <Cell key={r.name} fill={r.fill} />)}</Pie>
    <Tooltip contentStyle={tooltipStyle} formatter={(value, name, item) => [`${compact(Number(value))} · ${pct(item.payload.share)}`, String(name)]} />
  </PieChart></ResponsiveContainer><div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center"><span className="font-mono text-2xl text-[#211c17]">{compact(reviewCount ?? total)}</span><span className="text-[10px] uppercase tracking-widest text-[#756a5f]">reviews</span></div>
  <div className="mt-[-14px] flex justify-center gap-3 text-[10px] capitalize text-[#756a5f]">{rows.map((r) => <span key={r.name}><i className="mr-1 inline-block h-2 w-2" style={{ background: r.fill }} />{r.name}</span>)}</div></> : <Empty />}</div></ChartFrame>;
}

type AspectMetric = {
  category: string;
  review_count: number;
  sentiment_distribution: Record<'positive' | 'neutral' | 'negative', { review_count: number }>;
};
export function AspectSentiment({ data = [] }: { data?: AspectMetric[] }) {
  const rows = aspectChartRows(data).filter((r) => r.review_count > 0);
  return <ChartFrame title="Aspect sentiment" subtitle="Normalized widths; a review may mention more than one sentiment"><div className="h-64">{rows.length ? <ResponsiveContainer><BarChart data={rows} layout="vertical" margin={{ left: 20, right: 12, top: 2, bottom: 2 }}>
    <CartesianGrid stroke={grid} horizontal={false} /><XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={axis} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="category" width={82} tick={axis} axisLine={false} tickLine={false} />
    <Tooltip contentStyle={tooltipStyle} formatter={(v, key, item) => [`${compact(item.payload[key as 'positive'|'neutral'|'negative'])} reviews · ${Number(v).toFixed(1)}% visual share`, String(key)]} labelFormatter={(label, items) => `${label} · ${items[0]?.payload.review_count ?? 0} reviews${items[0]?.payload.overlap ? ' · mixed sentiment overlap' : ''}`} />
    <Bar dataKey="positive_pct" name="positive" stackId="a" fill={SENTIMENT_COLORS.positive} /><Bar dataKey="neutral_pct" name="neutral" stackId="a" fill={SENTIMENT_COLORS.neutral} /><Bar dataKey="negative_pct" name="negative" stackId="a" fill={SENTIMENT_COLORS.negative} />
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}

export function TopIssues({ data = [], onSelectIssue, limit = 8 }: { data?: MetricIssue[]; onSelectIssue?: (issue: MetricIssue) => void; limit?: number }) {
  const rows = useMemo(() => [...data].filter((x) => x.review_count > 0).sort((a, b) => b.review_count - a.review_count).slice(0, limit).map((x) => ({ ...x, label: shortLabel(x.canonical_name || x.canonical_id, 21) })), [data, limit]);
  return <ChartFrame title="Top issues" subtitle="Click a bar to inspect its evidence"><div className="h-72">{rows.length ? <ResponsiveContainer><BarChart data={rows} layout="vertical" margin={{ left: 8, right: 26, top: 0, bottom: 0 }}>
    <CartesianGrid stroke={grid} horizontal={false} /><XAxis type="number" allowDecimals={false} tick={axis} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="label" width={138} tick={axis} axisLine={false} tickLine={false} />
    <Tooltip contentStyle={tooltipStyle} labelFormatter={(_label, items) => items[0]?.payload.canonical_name || items[0]?.payload.canonical_id || ''} formatter={(v, _key, item) => [`${compact(Number(v))} · ${pct(item.payload.share_of_successful_reviews)}`, 'Reviews']} />
    <Bar dataKey="review_count" fill="#e86f21" maxBarSize={24} cursor={onSelectIssue ? 'pointer' : 'default'} onClick={(entry) => { const row = (entry as unknown as { payload?: MetricIssue }).payload; if (row) onSelectIssue?.(row); }}>{rows.map((row, i) => <Cell key={row.canonical_id} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}</Bar>
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}

export function IssueImpactScatter({ data = [], onSelectIssue }: { data?: MetricIssue[]; onSelectIssue?: (issue: MetricIssue) => void }) {
  const rows = issueImpactRows(data).filter((r) => r.frequency != null && r.average_rating != null);
  return <ChartFrame title="Issue impact" subtitle="Frequency versus average rating · larger, redder points have more 1–2★ reviews"><div className="h-72">{rows.length ? <ResponsiveContainer><ScatterChart margin={{ left: 2, right: 14, top: 8, bottom: 5 }}>
    <CartesianGrid stroke={grid} /><XAxis type="number" dataKey="frequency" name="Share" domain={[0, 'dataMax']} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} tick={axis} label={{ value: 'Issue share', position: 'insideBottom', offset: -1, fill: '#756a5f', fontSize: 11 }} />
    <YAxis type="number" dataKey="average_rating" name="Average rating" domain={[1, 5]} tick={axis} label={{ value: 'Avg rating', angle: -90, position: 'insideLeft', fill: '#756a5f', fontSize: 11 }} />
    <ZAxis type="number" dataKey="size" range={[50, 480]} /><ReferenceLine y={3} stroke="#bd8c65" strokeDasharray="4 4" />
    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={tooltipStyle} content={({ active, payload }) => {
      const p = payload?.[0]?.payload;
      if (!active || !p) return null;
      return <div className="border border-[#ddd1c3] bg-[#fffcf6] px-3 py-2 text-xs text-[#211c17]">
        <p className="mb-1 max-w-64 font-semibold">{p.name}</p>
        <p>{compact(Number(p.review_count))} reviews · {pct(p.frequency)}</p>
        <p>Average rating: {rating(p.average_rating)}</p>
        <p>1–2★ rating share: {pct(p.negativeShare)}</p>
      </div>;
    }} />
    <Scatter data={rows} name="Issues" onClick={(entry) => { const point = (entry as { payload?: { canonical_id?: string } }).payload; const issue = data.find((x) => x.canonical_id === point?.canonical_id); if (issue) onSelectIssue?.(issue); }} cursor={onSelectIssue ? 'pointer' : 'default'}>{rows.map((r) => <Cell key={r.canonical_id} fill={r.negativeShare != null && r.negativeShare >= 0.5 ? '#c74c36' : '#e86f21'} fillOpacity={0.8} />)}</Scatter>
  </ScatterChart></ResponsiveContainer> : <Empty />}</div><p className="mt-2 text-[10px] text-[#756a5f]">Based on the collected App Store review sample.</p></ChartFrame>;
}

export interface FeatureMetric { canonical_id: string; canonical_name: string | null; category?: string | null; review_count: number; share_of_successful_reviews: number | null }
export function FeatureRequests({ data = [], limit = 8 }: { data?: FeatureMetric[]; limit?: number }) {
  const rows = [...data].filter((r) => r.review_count > 0).sort((a, b) => b.review_count - a.review_count).slice(0, limit).map((r) => ({ ...r, label: shortLabel(r.canonical_name || r.canonical_id, 22) }));
  return <ChartFrame title="Feature requests" subtitle="Requests mentioned in analyzed reviews"><div className="h-64">{rows.length ? <ResponsiveContainer><BarChart data={rows} layout="vertical" margin={{ left: 12, right: 18, top: 0, bottom: 0 }}>
    <CartesianGrid stroke={grid} horizontal={false} /><XAxis type="number" allowDecimals={false} tick={axis} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="label" width={150} tick={axis} axisLine={false} tickLine={false} />
    <Tooltip contentStyle={tooltipStyle} labelFormatter={(_label, items) => items[0]?.payload.canonical_name || items[0]?.payload.canonical_id || ''} formatter={(v, _k, item) => [`${compact(Number(v))} · ${pct(item.payload.share_of_successful_reviews)}`, 'Reviews']} /><Bar dataKey="review_count" fill="#d8a241" maxBarSize={23} />
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}

function BreakdownBar({ title, labelKey, data = [] }: { title: string; labelKey: 'country' | 'language'; data?: DistributionRow[] }) {
  const rows = [...data].filter((r) => r.review_count > 0).sort((a, b) => b.review_count - a.review_count).slice(0, 12).map((r) => ({ ...r, label: r[labelKey] || 'Unknown' }));
  return <ChartFrame title={title}><div className="h-64">{rows.length ? <ResponsiveContainer><BarChart data={rows} layout="vertical" margin={{ left: 10, right: 14, top: 0, bottom: 0 }}>
    <CartesianGrid stroke={grid} horizontal={false} /><XAxis type="number" allowDecimals={false} tick={axis} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="label" width={70} tick={axis} axisLine={false} tickLine={false} />
    <Tooltip contentStyle={tooltipStyle} formatter={(v, _k, item) => [`${compact(Number(v))} reviews · ${rating(item.payload.average_rating)}`, 'Sample']} /><Bar dataKey="review_count" fill="#e86f21" maxBarSize={20} />
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}
export function ReviewsByCountry({ data = [] }: { data?: DistributionRow[] }) { return <BreakdownBar title="Reviews by country" labelKey="country" data={data} />; }
export function LanguageDistribution({ data = [] }: { data?: DistributionRow[] }) { return <BreakdownBar title="Language distribution" labelKey="language" data={data} />; }

export function AverageRatingByCountry({ data = [] }: { data?: DistributionRow[] }) {
  const rows = data.filter((r) => r.country && r.average_rating != null && r.review_count >= 5).sort((a, b) => b.review_count - a.review_count).slice(0, 12);
  return <ChartFrame title="Average rating by country" subtitle="Sample size is shown in each tooltip"><div className="h-60">{rows.length ? <ResponsiveContainer><BarChart data={rows} margin={{ left: -18, right: 8, top: 8, bottom: 4 }}>
    <CartesianGrid stroke={grid} vertical={false} /><XAxis dataKey="country" tick={axis} axisLine={false} tickLine={false} /><YAxis domain={[0, 5]} tick={axis} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} formatter={(v, _k, item) => [`${rating(Number(v))} · n=${item.payload.review_count}`, 'Average']} />
    <Bar dataKey="average_rating" fill="#4f7a59" maxBarSize={34} />
  </BarChart></ResponsiveContainer> : <Empty />}</div></ChartFrame>;
}

export function RatingByVersion({ data = [] }: { data?: DistributionRow[] }) {
  const rows = [...data].filter((r) => r.app_version && r.review_count > 0).sort((a, b) => a.app_version!.localeCompare(b.app_version!, undefined, { numeric: true }));
  const chartRows = rows.filter((row) => row.review_count >= 3 && row.average_rating != null);
  return <ChartFrame title="Rating by app version" subtitle={chartRows.length >= 3 ? 'Version samples with at least 3 reviews each' : 'Sparse version groups shown as a table'}>
    {chartRows.length >= 3 ? <div className="h-60"><ResponsiveContainer><ComposedChart data={chartRows} margin={{ left: -18, right: 8, top: 8, bottom: 4 }}><CartesianGrid stroke={grid} vertical={false} /><XAxis dataKey="app_version" tick={axis} axisLine={false} tickLine={false} /><YAxis yAxisId="rating" domain={[0, 5]} tick={axis} axisLine={false} tickLine={false} /><YAxis yAxisId="count" orientation="right" hide /><Tooltip contentStyle={tooltipStyle} formatter={(v, key, item) => [key === 'Rating' ? rating(Number(v)) : `${compact(Number(v))} reviews`, `${String(key)} · n=${item.payload.review_count}`]} /><Bar yAxisId="count" dataKey="review_count" name="Sample" fill="#eadbca" maxBarSize={30} /><Line yAxisId="rating" dataKey="average_rating" name="Rating" stroke="#e86f21" strokeWidth={2} dot={{ r: 3 }} connectNulls={false} /></ComposedChart></ResponsiveContainer></div> : rows.length ? <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead className="border-b border-[#ddd1c3] text-[#756a5f]"><tr><th className="py-2">Version</th><th>Reviews</th><th>Average rating</th></tr></thead><tbody>{rows.map((r) => <tr key={r.app_version} className="border-b border-[#eee6dc]"><td className="py-2 font-mono">{r.app_version}</td><td>{compact(r.review_count)}</td><td>{rating(r.average_rating)}</td></tr>)}</tbody></table></div> : <Empty />}
  </ChartFrame>;
}

export function IssuesOverTime({ timeBreakdown = {}, issues = [] }: { timeBreakdown?: Record<string, { issue_frequency?: Record<string, number>; issue_share?: Record<string, number | null> }>; issues?: MetricIssue[] }) {
  const rows = issuesOverTimeRows(timeBreakdown, issues); const top = [...issues].filter((x) => x.review_count > 0).sort((a, b) => b.review_count - a.review_count).slice(0, 5);
  return <ChartFrame title="Issues over time" subtitle="Review counts by month; sample history does not establish population trends"><div className="h-64">{rows.length && top.length ? <ResponsiveContainer><LineChart data={rows} margin={{ left: -18, right: 8, top: 8, bottom: 0 }}><CartesianGrid stroke={grid} /><XAxis dataKey="period" tick={axis} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tick={axis} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><>{top.map((issue, i) => <Line key={issue.canonical_id} dataKey={issue.canonical_id} name={issue.canonical_name || issue.canonical_id} stroke={CHART_COLORS[i % CHART_COLORS.length]} strokeWidth={2} dot={false} connectNulls />)}</></LineChart></ResponsiveContainer> : <Empty>Not enough data</Empty>}</div></ChartFrame>;
}

// Page-level adapters keep Overview wiring concise and ensure every chart reads
// the backend's response fields directly.
type MetricProps = { metrics: MetricsResponse; onIssueClick?: (id: string) => void };
export function RatingDistributionChart({ metrics }: MetricProps) { return <RatingDistribution data={metrics.basic.rating_distribution} />; }
export function SentimentDistributionChart({ metrics }: MetricProps) { return <SentimentDistribution distribution={metrics.sentiment.distribution} reviewCount={metrics.sentiment.sample_size} />; }
export function AspectSentimentChart({ metrics }: MetricProps) { return <AspectSentiment data={metrics.nlp?.aspect_metrics} />; }
export function TopIssuesChart({ metrics, onIssueClick }: MetricProps) { return <TopIssues data={metrics.nlp?.issue_metrics.issues} onSelectIssue={(issue) => onIssueClick?.(issue.canonical_id)} />; }
export function IssueImpactScatterChart({ metrics, onIssueClick }: MetricProps) { return <IssueImpactScatter data={metrics.nlp?.issue_metrics.issues} onSelectIssue={(issue) => onIssueClick?.(issue.canonical_id)} />; }
export function FeatureRequestsChart({ metrics }: MetricProps) { return <FeatureRequests data={metrics.nlp?.feature_request_metrics.feature_requests} />; }
export function ReviewsByCountryChart({ metrics }: MetricProps) { return <ReviewsByCountry data={metrics.basic.country_statistics} />; }
export function LanguageDistributionChart({ metrics }: MetricProps) { return <LanguageDistribution data={metrics.basic.language_statistics} />; }
export function AverageRatingByCountryChart({ metrics }: MetricProps) { return <AverageRatingByCountry data={metrics.basic.country_statistics} />; }
export function RatingByVersionChart({ metrics }: MetricProps) { return <RatingByVersion data={metrics.basic.rating_by_version} />; }
export function IssuesOverTimeChart({ metrics }: MetricProps) { return <IssuesOverTime timeBreakdown={metrics.nlp?.time_breakdown} issues={metrics.nlp?.issue_metrics.issues} />; }

export { ratingChartRows, sentimentChartRows, aspectChartRows, issueImpactRows, issuesOverTimeRows } from './chartData';
