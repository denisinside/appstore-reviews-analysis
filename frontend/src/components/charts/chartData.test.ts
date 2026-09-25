import { describe, expect, it } from 'vitest'
import { aspectChartRows, issueImpactRows, issuesOverTimeRows, ratingChartRows } from './chartData'

describe('chart presentation transforms', () => {
  it('keeps missing rating buckets visible without inventing shares', () => {
    const rows = ratingChartRows([{ rating: 5, count: 7, share: 1 }])
    expect(rows).toHaveLength(5)
    expect(rows[0]).toEqual({ rating: '1★', count: 0, share: null })
    expect(rows[4]).toEqual({ rating: '5★', count: 7, share: 1 })
  })

  it('normalizes overlapping aspect classes only for visual widths', () => {
    const [row] = aspectChartRows([{ category: 'stability', review_count: 2, sentiment_distribution: {
      positive: { review_count: 1 }, neutral: { review_count: 0 }, negative: { review_count: 2 },
    } }])
    expect(row.overlap).toBe(true)
    expect(row.negative).toBe(2)
    expect(row.positive_pct + row.neutral_pct + row.negative_pct).toBeCloseTo(100)
  })

  it('preserves backend issue frequency and null rating for scatter filtering', () => {
    const [row] = issueImpactRows([{ canonical_id: 'one', canonical_name: 'Issue', review_count: 4, share_of_successful_reviews: .2, average_rating: null, negative_rating_share: null }])
    expect(row.frequency).toBe(.2)
    expect(row.average_rating).toBeNull()
    expect(row.negativeShare).toBeNull()
  })

  it('uses backend time bucket counts without filling missing values with zero', () => {
    const issues = [{ canonical_id: 'one', canonical_name: 'Issue', review_count: 4, share_of_successful_reviews: .2 }]
    const rows = issuesOverTimeRows({ '2026-01': { successful_review_count: 6, issue_frequency: { one: 2 } }, '2026-02': { successful_review_count: 2, issue_frequency: {} }, '2026-03': { successful_review_count: 7, issue_frequency: { one: 1 } } }, issues)
    expect(rows).toEqual([{ period: '2026-01', one: 2 }, { period: '2026-03', one: 1 }])
  })
})
