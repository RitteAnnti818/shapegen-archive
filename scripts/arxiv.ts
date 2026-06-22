import { XMLParser } from 'fast-xml-parser';
import type { ArxivCandidate } from './schema';

const API = 'http://export.arxiv.org/api/query';

/** Query families narrowed to **3D shape generation** (object/scene geometry synthesis). */
export const QUERY_FAMILIES: string[] = [
  // -- core 3D shape generation --
  'ti:"3D shape generation"',
  'abs:"3D shape generation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"3D generation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"3D object generation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"shape synthesis" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"3D content creation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"3D asset generation"',
  'abs:"generative 3D" AND (cat:cs.CV OR cat:cs.GR)',

  // -- text/image conditioned 3D --
  'ti:"text-to-3D"',
  'abs:"text-to-3D" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"image-to-3D" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"single-image" AND abs:"3D generation"',
  'abs:"score distillation" AND abs:3D',
  'abs:"multi-view diffusion" AND (cat:cs.CV OR cat:cs.GR)',

  // -- 3D diffusion / flow --
  'abs:"3D diffusion" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"diffusion" AND abs:"shape generation"',
  'abs:"latent diffusion" AND abs:3D AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"flow matching" AND abs:"3D shape"',
  'abs:"rectified flow" AND abs:3D AND (cat:cs.CV OR cat:cs.GR)',

  // -- representation: mesh / point cloud / voxel --
  'abs:"mesh generation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"point cloud generation" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"voxel" AND abs:"shape generation"',
  'abs:"autoregressive" AND abs:mesh AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"mesh diffusion"',

  // -- representation: implicit / SDF / neural field --
  'abs:"implicit" AND abs:"shape generation"',
  'abs:"signed distance" AND abs:generation AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"SDF" AND abs:"3D generation"',
  'abs:"neural field" AND abs:generation AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"occupancy" AND abs:"shape generation"',

  // -- representation: gaussian / triplane / NeRF generative --
  'abs:"3D gaussian" AND abs:generation AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"gaussian splatting" AND abs:generation AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"triplane" AND abs:generation',
  'abs:"large reconstruction model" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"feed-forward" AND abs:"3D generation"',

  // -- generative model families --
  'abs:"3D GAN" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"3D-aware" AND abs:generation AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"variational autoencoder" AND abs:"3D shape"',
  'abs:"vector quantized" AND abs:"3D shape"',

  // -- editing / part / structure-aware --
  'abs:"shape editing" AND (cat:cs.CV OR cat:cs.GR)',
  'abs:"part-based" AND abs:"shape generation"',
  'abs:"structure-aware" AND abs:"3D shape"',
  'abs:"compositional" AND abs:"3D generation"',
];

export interface FetchOptions {
  fromDate: string; // YYYYMMDD
  toDate?: string; // YYYYMMDD
  maxPerQuery?: number;
}

export async function fetchArxivForQuery(
  query: string,
  opts: FetchOptions,
): Promise<ArxivCandidate[]> {
  const to = opts.toDate ?? todayYmd();
  const ranged = `(${query}) AND submittedDate:[${opts.fromDate}0000 TO ${to}2359]`;
  const params = new URLSearchParams({
    search_query: ranged,
    start: '0',
    max_results: String(opts.maxPerQuery ?? 100),
    sortBy: 'submittedDate',
    sortOrder: 'descending',
  });
  const url = `${API}?${params.toString()}`;

  let lastErr: Error | null = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    if (attempt > 0) await sleep(5000 * Math.pow(2, attempt - 1));
    try {
      const res = await fetch(url, {
        headers: { 'User-Agent': 'hairxiv/0.1 (research crawl)' },
      });
      if (res.status === 429 || res.status === 503) {
        lastErr = new Error(`HTTP ${res.status}`);
        continue;
      }
      if (!res.ok) throw new Error(`arxiv HTTP ${res.status}`);
      const xml = await res.text();
      return parseFeed(xml);
    } catch (err) {
      lastErr = err as Error;
    }
  }
  throw lastErr ?? new Error('arxiv: unknown failure');
}

export async function fetchAllFamilies(opts: FetchOptions): Promise<ArxivCandidate[]> {
  const map = new Map<string, ArxivCandidate>();
  for (let i = 0; i < QUERY_FAMILIES.length; i++) {
    const q = QUERY_FAMILIES[i];
    try {
      const rows = await fetchArxivForQuery(q, opts);
      let added = 0;
      for (const r of rows) {
        if (!map.has(r.id)) {
          map.set(r.id, r);
          added++;
        }
      }
      console.log(
        `[arxiv] (${i + 1}/${QUERY_FAMILIES.length}) ${rows.length} hits, ${added} new · ${q}`,
      );
    } catch (err) {
      console.error(`[arxiv] query "${q}" failed:`, (err as Error).message);
    }
    if (i < QUERY_FAMILIES.length - 1) await sleep(6000);
  }
  return [...map.values()].sort((a, b) => b.published.localeCompare(a.published));
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function todayYmd(): string {
  const d = new Date();
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}`;
}
function pad(n: number) {
  return String(n).padStart(2, '0');
}

function parseFeed(xml: string): ArxivCandidate[] {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '',
    isArray: (name) =>
      name === 'entry' || name === 'author' || name === 'category' || name === 'link',
  });
  const feed = parser.parse(xml)?.feed;
  if (!feed || !feed.entry) return [];
  return feed.entry.map((e: unknown) => toCandidate(e as Record<string, unknown>));
}

function toCandidate(e: Record<string, unknown>): ArxivCandidate {
  const idUrl = (e.id as string) ?? '';
  const m = idUrl.match(/abs\/([^v]+)(v\d+)?$/);
  const id = m?.[1] ?? idUrl;
  const version = m?.[2] ?? 'v1';
  const authors = ((e.author as Array<{ name?: string }>) ?? [])
    .map((a) => (a?.name ?? '').trim())
    .filter(Boolean);
  const links = (e.link as Array<{ href?: string; title?: string; type?: string }>) ?? [];
  const pdf = links.find((l) => l.title === 'pdf' || l.type === 'application/pdf')?.href;
  const cats =
    ((e.category as Array<{ term?: string }>) ?? [])
      .map((c) => c.term)
      .filter((x): x is string => !!x) ?? [];
  const primaryTerm =
    ((e['arxiv:primary_category'] as { term?: string }) ?? {}).term ?? cats[0] ?? 'cs.CV';
  return {
    id,
    version,
    url: idUrl.replace(/v\d+$/, ''),
    title: String(e.title ?? '').trim().replace(/\s+/g, ' '),
    authors,
    abstract: String(e.summary ?? '').trim().replace(/\s+/g, ' '),
    published: String(e.published ?? ''),
    updated: String(e.updated ?? ''),
    primaryCategory: primaryTerm,
    categories: cats,
    pdfUrl: pdf,
  };
}
