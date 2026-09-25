/** Presentation-only transforms for deterministic API metrics. */
export interface RatingBucket { rating: number; count: number; share: number | null }
export interface MetricIssue {
  canonical_id: string; canonical_name: string | null; category?: string | null;
  review_count: number; share_of_successful_reviews: number | null;
  average_rating?: number | null; negative_rating_share?: number | null; rating_sample_size?: number;
}
export interface DistributionRow {
  country?: string | null; language?: string | null; app_version?: string | null;
  review_count: number; average_rating: number | null;
}
export const SENTIMENT_COLORS = { positive: '#4f7a59', neutral: '#d8c6aa', negative: '#c74c36' } as const;
export const CHART_COLORS = ['#e86f21', '#c74c36', '#d8a241', '#4f7a59', '#786b60', '#bd8c65'];

export function ratingChartRows(rows: RatingBucket[] = []) {
  return [1, 2, 3, 4, 5].map((rating) => {
    const row = rows.find((item) => item.rating === rating);
    return { rating: `${rating}★`, count: row?.count ?? 0, share: row?.share ?? null };
  });
}
export function sentimentChartRows(distribution: Partial<Record<'positive' | 'neutral' | 'negative', { count: number; share: number | null }>> = {}) {
  return (['positive', 'neutral', 'negative'] as const).map((sentiment) => ({ name: sentiment, count: distribution[sentiment]?.count ?? 0, share: distribution[sentiment]?.share ?? null, fill: SENTIMENT_COLORS[sentiment] }));
}
export function aspectChartRows(rows: Array<{ category: string; review_count: number; sentiment_distribution: Record<'positive' | 'neutral' | 'negative', { review_count: number }> }> = []) {
  return rows.map((row) => {
    const positive = row.sentiment_distribution?.positive?.review_count ?? 0;
    const neutral = row.sentiment_distribution?.neutral?.review_count ?? 0;
    const negative = row.sentiment_distribution?.negative?.review_count ?? 0;
    const total = positive + neutral + negative;
    // Mixed reviews can occur in multiple classes; normalize only the visual widths.
    return { category: row.category, review_count: row.review_count, positive, neutral, negative,
      positive_pct: total ? positive / total * 100 : 0, neutral_pct: total ? neutral / total * 100 : 0,
      negative_pct: total ? negative / total * 100 : 0, overlap: total > row.review_count };
  });
}
export function issueImpactRows(issues: MetricIssue[] = []) {
  return issues.filter((row) => row.review_count > 0).map((row) => ({ ...row,
    name: row.canonical_name || row.canonical_id, frequency: row.share_of_successful_reviews,
    negativeShare: row.negative_rating_share, size: Math.max(40, Math.min(480, row.review_count * 24)),
  }));
}
export function issuesOverTimeRows(timeBreakdown: Record<string, { successful_review_count?: number; issue_frequency?: Record<string, number>; issue_share?: Record<string, number | null> }> = {}, issues: MetricIssue[] = []) {
  const periods = Object.keys(timeBreakdown).filter((period) => (timeBreakdown[period].successful_review_count ?? 0) >= 5).sort();
  if (periods.length < 2) return [];
  const top = [...issues].filter((issue) => issue.review_count > 0).sort((a, b) => b.review_count - a.review_count).slice(0, 5);
  return periods.map((period) => {
    const point: Record<string, string | number | null> = { period };
    for (const issue of top) point[issue.canonical_id] = timeBreakdown[period].issue_frequency?.[issue.canonical_id] ?? null;
    return point;
  });
}
