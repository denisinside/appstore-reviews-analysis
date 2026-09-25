import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, ArrowRight, Search, X } from 'lucide-react';
import { getReviews } from '../api/client';
import type { Review, ReviewsPage } from '../api/types';

type Sentiment = 'positive' | 'neutral' | 'negative';
type Props = {
  scanId: string;
  focusReviewIds?: string[];
  focusReviewId?: string;
  onCloseFocus?: () => void;
};

const PAGE_SIZE = 25;
const FOCUS_PAGE_SIZE = 500;
const sentimentTone: Record<Sentiment, string> = {
  positive: 'bg-[#e8efe5] text-[#426849]',
  neutral: 'bg-[#f1eadc] text-[#806b48]',
  negative: 'bg-[#f7e7e1] text-[#a84936]',
};

function pretty(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.replaceAll('_', ' ') : '—';
}

function ReviewDetail({ review, onClose }: { review: Review; onClose: () => void }) {
  const analysis = review.issue_analysis;
  const aspects = Array.isArray(analysis?.aspects) ? analysis.aspects : [];
  const signals = (key: 'issues' | 'feature_requests') => aspects.flatMap((aspect) =>
    Array.isArray(aspect[key]) ? aspect[key].map((signal) => ({
      description: typeof signal.description === 'string' ? signal.description : undefined,
      evidence: typeof signal.evidence === 'string' ? signal.evidence : undefined,
      category: aspect.category,
      aspect: typeof aspect.aspect === 'string' ? aspect.aspect : undefined,
    })) : [],
  );
  const issues = signals('issues');
  const requests = signals('feature_requests');
  const keywords = Array.isArray(review.keywords) ? review.keywords : [];

  return <>
    <button aria-label="Close review detail" onClick={onClose} className="fixed inset-0 z-40 cursor-default bg-[#211c17]/25" />
    <aside className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-[#ddd1c3] bg-[#fffcf6] shadow-2xl">
      <header className="flex items-start justify-between border-b border-[#e5dbcf] px-6 py-5">
        <div>
          <p className="font-mono text-xs uppercase tracking-[.16em] text-[#8c7e70]">Review detail · {review.review_id}</p>
          <h2 className="mt-2 text-xl font-semibold text-[#211c17]">{review.rating ?? '—'} <span className="text-[#e5a536]">★</span> <span className="font-normal">{review.title || 'Untitled review'}</span></h2>
        </div>
        <button onClick={onClose} className="rounded-sm p-2 text-[#756a5f] hover:bg-[#f3ede4]" aria-label="Close"><X size={18} /></button>
      </header>
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mb-6 flex flex-wrap gap-x-5 gap-y-2 text-xs text-[#756a5f]">
          <span>{pretty(review.country)}</span><span>{pretty(review.language)}</span><span>Version {pretty(review.app_version)}</span><span>{review.updated_at ? new Date(review.updated_at).toLocaleDateString() : 'Date unavailable'}</span>
          {review.sentiment && <span className={`rounded-sm px-2 py-1 font-medium capitalize ${sentimentTone[review.sentiment]}`}>{review.sentiment}</span>}
        </div>
        <p className="whitespace-pre-wrap text-[15px] leading-7 text-[#342d26]">{review.text || 'No review text was provided.'}</p>
        <DetailSection title="Keywords">{keywords.length ? <div className="flex flex-wrap gap-2">{keywords.map((item, index) => <span key={`${item.text}-${index}`} className="rounded-sm border border-[#e4d9cc] px-2 py-1 text-xs text-[#695c50]">{item.text}</span>)}</div> : <EmptyLine />}</DetailSection>
        <DetailSection title="Aspects">{aspects.length ? <div className="space-y-2">{aspects.map((aspect, index) => <div key={`${aspect.category}-${aspect.aspect}-${index}`} className="flex justify-between gap-3 border-b border-[#eee6dc] py-2 text-sm"><span className="capitalize">{pretty(aspect.category)} <span className="text-[#8c7e70]">/ {pretty(aspect.aspect)}</span></span><span className="text-xs capitalize text-[#756a5f]">{pretty(aspect.sentiment)}</span></div>)}</div> : <EmptyLine />}</DetailSection>
        <DetailSection title="Issues">{issues.length ? <SignalList items={issues} /> : <EmptyLine />}</DetailSection>
        <DetailSection title="Feature requests">{requests.length ? <SignalList items={requests} /> : <EmptyLine />}</DetailSection>
      </div>
    </aside>
  </>;
}

function DetailSection({ title, children }: { title: string; children: ReactNode }) {
  return <section className="mt-7"><h3 className="mb-3 font-mono text-[11px] font-semibold uppercase tracking-[.16em] text-[#8c7e70]">{title}</h3>{children}</section>;
}
function EmptyLine() { return <p className="text-sm text-[#a69a8d]">No data available</p>; }
function SignalList({ items }: { items: Array<{ description?: string; evidence?: string; category?: string; aspect?: string }> }) {
  return <ul className="space-y-3">{items.map((item, index) => <li key={`${item.description}-${index}`} className="border-l-2 border-[#e86f21] pl-3"><p className="text-sm text-[#342d26]">{item.description || 'Unlabeled signal'}</p>{item.evidence && <p className="mt-1 text-xs italic text-[#8c7e70]">“{item.evidence}”</p>}</li>)}</ul>;
}

export default function ReviewsExplorer({ scanId, focusReviewIds, focusReviewId, onCloseFocus }: Props) {
  const [search, setSearch] = useState('');
  const [rating, setRating] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [country, setCountry] = useState('');
  const [language, setLanguage] = useState('');
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Review | null>(null);
  const lastAutoOpenedFocus = useRef('');

  const query = useQuery<ReviewsPage>({
    queryKey: ['reviews', scanId, offset, search, rating, sentiment, country, language],
    queryFn: () => getReviews(scanId, {
      offset, limit: PAGE_SIZE, q: search.trim() || undefined,
      rating: rating ? Number(rating) : undefined,
      sentiment: sentiment ? sentiment as Sentiment : undefined,
      country: country.trim() || undefined, language: language.trim() || undefined,
    }),
    placeholderData: (previous) => previous,
  });
  const focusIds = useMemo(() => new Set([...(focusReviewIds || []), ...(focusReviewId ? [focusReviewId] : [])].map(String)), [focusReviewIds, focusReviewId]);
  const focusQuery = useQuery({
    queryKey: ['reviews-focus', scanId, [...focusIds].sort().join(',')],
    enabled: focusIds.size > 0,
    queryFn: async () => {
      const found = new Map<string, Review>();
      // The API has no review ID lookup. Fetch paginated slices only when a
      // supporting-review link is opened; one slice is the backend's max limit.
      for (let pageOffset = 0; pageOffset < FOCUS_PAGE_SIZE * 32; pageOffset += FOCUS_PAGE_SIZE) {
        const result = await getReviews(scanId, { offset: pageOffset, limit: FOCUS_PAGE_SIZE });
        for (const row of result.items) if (focusIds.has(String(row.review_id))) found.set(String(row.review_id), row);
        if (found.size === focusIds.size || pageOffset + FOCUS_PAGE_SIZE >= result.total || result.items.length === 0) break;
      }
      return [...found.values()];
    },
  });
  const items = focusIds.size ? focusQuery.data || [] : query.data?.items || [];
  const total = focusIds.size ? items.length : query.data?.total ?? 0;
  const page = focusIds.size ? 1 : Math.floor(offset / PAGE_SIZE) + 1;
  const pages = focusIds.size ? 1 : Math.max(1, Math.ceil(total / PAGE_SIZE));
  const focusKey = [...focusIds].sort().join(',');
  useEffect(() => {
    if (focusKey && focusQuery.data?.length && lastAutoOpenedFocus.current !== focusKey) {
      setSelected(focusQuery.data[0]);
      lastAutoOpenedFocus.current = focusKey;
    }
  }, [focusKey, focusQuery.data]);
  const updateFilter = (setter: (value: string) => void) => (value: string) => { setter(value); setOffset(0); };

  return <div className="text-[#211c17]">
    {focusIds.size > 0 && <div className="mb-5 flex items-center justify-between border border-[#f0c99f] bg-[#fff5e8] px-4 py-3 text-sm"><span>{focusQuery.isPending ? 'Finding supporting reviews…' : `${items.length} of ${focusIds.size} linked review${focusIds.size === 1 ? '' : 's'} found.`}</span><button className="font-medium text-[#b94d0d] underline" onClick={onCloseFocus}>Clear focus</button></div>}
    <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
      <label className="relative sm:col-span-2 xl:col-span-1"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#8c7e70]" /><input aria-label="Search reviews" value={search} onChange={(e) => updateFilter(setSearch)(e.target.value)} placeholder="Search reviews" className="w-full border border-[#ddd1c3] bg-[#fffcf6] py-2.5 pl-9 pr-3 text-sm outline-none focus:border-[#e86f21]" /></label>
      <select aria-label="Filter by rating" value={rating} onChange={(e) => updateFilter(setRating)(e.target.value)} className="border border-[#ddd1c3] bg-[#fffcf6] px-3 py-2.5 text-sm"><option value="">All ratings</option>{[5, 4, 3, 2, 1].map((n) => <option key={n} value={n}>{n} {n === 1 ? 'star' : 'stars'}</option>)}</select>
      <select aria-label="Filter by sentiment" value={sentiment} onChange={(e) => updateFilter(setSentiment)(e.target.value)} className="border border-[#ddd1c3] bg-[#fffcf6] px-3 py-2.5 text-sm"><option value="">All sentiment</option>{['positive', 'neutral', 'negative'].map((v) => <option key={v} value={v}>{v}</option>)}</select>
      <input aria-label="Filter by country" value={country} onChange={(e) => updateFilter(setCountry)(e.target.value)} placeholder="Country code" className="border border-[#ddd1c3] bg-[#fffcf6] px-3 py-2.5 text-sm outline-none focus:border-[#e86f21]" />
      <input aria-label="Filter by language" value={language} onChange={(e) => updateFilter(setLanguage)(e.target.value)} placeholder="Language" className="border border-[#ddd1c3] bg-[#fffcf6] px-3 py-2.5 text-sm outline-none focus:border-[#e86f21]" />
    </div>
    {(focusIds.size ? focusQuery.isPending : query.isPending) ? <div className="border-y border-[#e5dbcf] py-14 text-center text-sm text-[#8c7e70]">Loading reviews…</div> : (focusIds.size ? focusQuery.isError : query.isError) ? <div role="alert" className="border border-[#e9c4b9] bg-[#fbefeb] p-5 text-sm text-[#93412f]">Could not load reviews: {(focusIds.size ? focusQuery.error : query.error) instanceof Error ? (focusIds.size ? focusQuery.error : query.error)?.message : 'Unexpected API error'}</div> : total === 0 ? <div className="border-y border-[#e5dbcf] py-14 text-center"><p className="font-medium">{focusIds.size ? 'Supporting reviews were not found in the collected sample' : 'No reviews match these filters'}</p><p className="mt-1 text-sm text-[#8c7e70]">{focusIds.size ? 'Clear focus to browse and filter the complete collection.' : 'Try changing your search or filters.'}</p></div> : <>
      <div className="overflow-x-auto border-y border-[#ddd1c3]">
        <table className="w-full min-w-[850px] border-collapse text-left">
          <thead><tr className="border-b border-[#ddd1c3] font-mono text-[10px] uppercase tracking-[.12em] text-[#8c7e70]"><th className="px-3 py-3 font-medium">Rating</th><th className="px-3 py-3 font-medium">Sentiment</th><th className="px-3 py-3 font-medium">Country</th><th className="px-3 py-3 font-medium">Language</th><th className="px-3 py-3 font-medium">Version</th><th className="w-[48%] px-3 py-3 font-medium">Review</th></tr></thead>
          <tbody>{items.map((review) => {
            const isFocused = focusIds.has(String(review.review_id));
            return <tr key={review.review_id} onClick={() => setSelected(review)} className={`cursor-pointer border-b border-[#eee6dc] align-top transition-colors hover:bg-[#f6f1e8] ${isFocused ? 'bg-[#fff4e4] ring-1 ring-inset ring-[#edb875]' : ''}`}>
              <td className="whitespace-nowrap px-3 py-4 font-mono text-sm">{review.rating ?? '—'} <span className="text-[#e5a536]">★</span></td>
              <td className="px-3 py-4">{review.sentiment ? <span className={`rounded-sm px-2 py-1 text-[11px] capitalize ${sentimentTone[review.sentiment]}`}>{review.sentiment}</span> : <span className="text-xs text-[#a69a8d]">—</span>}</td>
              <td className="px-3 py-4 text-xs uppercase text-[#695c50]">{pretty(review.country)}</td><td className="px-3 py-4 text-xs text-[#695c50]">{pretty(review.language)}</td><td className="px-3 py-4 font-mono text-xs text-[#695c50]">{pretty(review.app_version)}</td>
              <td className="px-3 py-4"><div className="text-sm font-medium text-[#342d26]">{review.title || 'Untitled review'}</div><p className="mt-1 line-clamp-2 text-xs leading-5 text-[#756a5f]">{review.text || 'No review text'}</p></td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 py-4 text-xs text-[#756a5f]"><span><b className="font-mono text-[#342d26]">{(focusIds.size ? focusIds.size : total).toLocaleString()}</b> {focusIds.size ? 'linked review IDs' : 'reviews'} · page {page} of {pages}</span>{!focusIds.size && <div className="flex items-center gap-2"><button onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} disabled={offset === 0 || query.isFetching} className="flex items-center gap-1 border border-[#ddd1c3] px-3 py-2 disabled:opacity-40"><ArrowLeft size={14} /> Previous</button><button onClick={() => setOffset(offset + PAGE_SIZE)} disabled={offset + PAGE_SIZE >= total || query.isFetching} className="flex items-center gap-1 border border-[#ddd1c3] px-3 py-2 disabled:opacity-40">Next <ArrowRight size={14} /></button></div>}</div>
    </>}
    {selected && <ReviewDetail review={selected} onClose={() => setSelected(null)} />}
  </div>;
}
