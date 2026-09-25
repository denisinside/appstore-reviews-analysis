export type ScanMode = "country" | "top";
export type AnalysisStatus = "not_started" | "queued" | "running" | "completed" | "failed";
export type SentimentLabel = "positive" | "neutral" | "negative";

export interface ScanOptions {
  countries: Array<{ code: string; name: string }>;
  max_pages: number;
  max_top_countries: number;
}

export interface CollectScanRequest {
  app_id: string;
  mode?: ScanMode;
  country?: string;
  top_n?: number;
  max_pages?: number;
  force_refresh?: boolean;
}

export type CreateScanRequest = CollectScanRequest;

export interface BasicMetricGroup {
  review_count: number;
  average_rating: number | null;
}

export interface RatingDistributionRow {
  rating: number;
  count: number;
  share: number | null;
}

export interface BasicMetrics {
  review_count: number;
  average_rating: number | null;
  rating_distribution: RatingDistributionRow[];
  negative_rating_share: number | null;
  country_statistics: Array<BasicMetricGroup & { country: string | null }>;
  language_statistics: Array<BasicMetricGroup & { language: string | null }>;
  rating_by_version: Array<BasicMetricGroup & { app_version: string | null }>;
}

export interface Scan {
  scan_id: string;
  app_id: string;
  app_name?: string | null;
  collection_mode: ScanMode;
  country: string | null;
  top_n: number | null;
  max_pages: number;
  collection_status: string | null;
  analysis_status: AnalysisStatus;
  progress?: {
    percent: number;
    stage: string;
    message: string;
    completed?: number | null;
    total?: number | null;
  } | null;
  review_count: number;
  created_at: string;
  updated_at: string;
  error: string | null;
  api_usage?: Record<string, unknown>;
  basic_metrics?: BasicMetrics;
  collection_errors?: Array<Record<string, unknown>>;
}

export interface ScanSummary {
  scan_id: string;
  app_id: string;
  app_name: string | null;
  created_at: string | null;
  analysis_status: AnalysisStatus;
  collection_mode: ScanMode;
  country: string | null;
  top_n: number | null;
  review_count: number;
  average_rating: number | null;
  negative_sentiment_share: number | null;
  issue_count: number | null;
}

export interface ScansResponse { items: ScanSummary[] }

export interface AnalyzeScanRequest {
  batch_size?: number;
  concurrency?: number;
  requests_per_minute?: number;
  max_cost_usd?: number | null;
  run_local_nlp?: boolean;
}

export interface AnalyzeScanResponse {
  scan_id: string;
  analysis_status: "queued";
}

export interface DistributionCount {
  count: number;
  share: number | null;
}

export interface SentimentSummary {
  sample_size: number;
  distribution: Record<SentimentLabel, DistributionCount>;
}

export interface CommonKeyword {
  phrase: string;
  normalized_phrase: string;
  review_count: number;
  share: number;
  average_score: number | null;
  example_review_ids: string[];
}

export interface AspectMetric {
  category: string;
  review_count: number;
  denominator_review_count: number;
  mention_count: number;
  share_of_successful_reviews: number | null;
  sentiment_distribution: Record<SentimentLabel, {
    review_count: number;
    mention_count: number;
    share_of_aspect_reviews: number | null;
  }>;
  mixed_review_count: number;
  negative_aspect_share: number | null;
}

export interface IssueMetric {
  canonical_id: string;
  canonical_name: string | null;
  category: string | null;
  review_count: number;
  denominator_review_count?: number;
  review_ids?: string[];
  share_of_successful_reviews: number | null;
  average_rating?: number | null;
  negative_rating_review_count?: number;
  negative_rating_share?: number | null;
  rating_sample_size?: number;
  sample_review_ids: string[];
}

export type FeatureRequestMetric = Omit<IssueMetric, "average_rating" | "negative_rating_review_count" | "negative_rating_share" | "rating_sample_size">

export interface IssueMetrics {
  reviews_with_any_issue_count: number;
  reviews_with_any_issue_share: number | null;
  issues: IssueMetric[];
}

export interface FeatureRequestMetrics {
  reviews_with_any_feature_request_count: number;
  reviews_with_any_feature_request_share: number | null;
  feature_requests: FeatureRequestMetric[];
}

export interface NlpMetrics {
  analysis_summary: {
    input_review_count: number;
    analyzed_review_count: number;
    analysis_error_count: number;
    rating_sample_size: number;
    average_rating: number | null;
    rules: string[];
    metric_rules: Record<string, string>;
  };
  aspect_metrics: AspectMetric[];
  issue_metrics: IssueMetrics;
  feature_request_metrics: FeatureRequestMetrics;
  coverage_metrics: Record<string, number | number[] | null>;
  country_breakdown: Record<string, BreakdownMetric>;
  version_breakdown: Record<string, BreakdownMetric>;
  time_breakdown: Record<string, BreakdownMetric>;
}

export interface BreakdownMetric {
  successful_review_count: number;
  denominator_review_count: number;
  issue_frequency: Record<string, number>;
  issue_share: Record<string, number | null>;
  aspect_sentiment: Record<string, Record<SentimentLabel, number>>;
}

export interface MetricsResponse {
  basic: BasicMetrics;
  sentiment: SentimentSummary;
  common_keywords_by_language: Record<string, CommonKeyword[]>;
  nlp: NlpMetrics | null;
}

export interface InsightMetrics {
  review_count: number;
  share_of_successful_reviews: number | null;
  average_rating?: number | null;
  negative_rating_share?: number | null;
  rating_sample_size?: number;
}

export interface IssueInsight {
  canonical_id: string;
  canonical_name: string | null;
  category: string | null;
  metrics: InsightMetrics;
  sample_size: number;
  caveat: string;
  finding: string;
  recommended_actions: string[];
  supporting_review_ids: string[];
  user_impact: string;
}

export interface FeatureRequestInsight {
  canonical_id: string;
  canonical_name: string | null;
  category: string | null;
  metrics: InsightMetrics;
  sample_size: number;
  caveat: string;
  finding: string;
  recommended_action: string;
  supporting_review_ids: string[];
}

export interface InsightsResponse {
  model: string;
  selection: Record<string, unknown>;
  analysis_summary: NlpMetrics["analysis_summary"];
  coverage_metrics: Record<string, number | number[] | null>;
  overall_summary: string;
  issue_insights: IssueInsight[];
  feature_request_insights: FeatureRequestInsight[];
  limitations: string[];
}

export interface IssueCatalogSource {
  source_id: string;
  aspect?: string;
  description?: string;
  descriptions?: string[];
  evidence?: string[];
  review_ids?: string[];
}

export interface CatalogEntry {
  canonical_id: string;
  canonical_name: string | null;
  category: string | null;
  source_issues?: IssueCatalogSource[];
  source_requests?: IssueCatalogSource[];
  review_ids?: string[];
  category_reconciliation?: string;
}

export interface IssuesResponse {
  issues: CatalogEntry[];
  feature_requests: CatalogEntry[];
}

export interface ReviewKeyword {
  text: string;
  score: number;
  evidence_span?: string;
  source_start?: number;
  source_end?: number;
}

export interface ReviewAspect {
  category: string;
  sentiment?: SentimentLabel | null;
  issues?: Array<Record<string, unknown>>;
  feature_requests?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface ReviewIssueAnalysis {
  status: string;
  aspects: ReviewAspect[];
  [key: string]: unknown;
}

export interface Review {
  review_id: string;
  app_id: string;
  country: string | null;
  observed_countries?: string[];
  title: string | null;
  text: string | null;
  rating: number;
  app_version: string | null;
  updated_at: string | null;
  language: string | null;
  language_confidence?: number | null;
  source?: Record<string, unknown>;
  sentiment: SentimentLabel | null;
  sentiment_scores: Record<SentimentLabel, number> | null;
  keywords: ReviewKeyword[];
  issue_analysis: ReviewIssueAnalysis | null;
}

export interface ReviewsQuery {
  offset?: number;
  limit?: number;
  rating?: number;
  sentiment?: SentimentLabel;
  language?: string;
  country?: string;
  q?: string;
}

export type ReviewFilters = ReviewsQuery;

export interface ReviewsResponse {
  total: number;
  offset: number;
  limit: number;
  items: Review[];
}

export type ReviewsPage = ReviewsResponse;

export interface DiscoveryCountry {
  country: string;
  rank: number | null;
  review_count: number | null;
  newest_review_at: string | null;
  oldest_review_at: string | null;
  status: "success" | "partial" | "failed";
}

export interface DiscoveryResponse {
  app_id: string;
  mode: "discovery";
  countries: DiscoveryCountry[];
  errors: Array<Record<string, unknown>>;
}
