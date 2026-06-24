import OpenAI from 'openai';
import type { ArxivCandidate, TaggedFields } from './schema';

const client = new OpenAI();
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

/**
 * System prompt narrowed to **3D shape generation** specifically.
 * Accepts generative synthesis of 3D geometry (mesh / point cloud / SDF / implicit /
 * gaussian / triplane) — text-to-3D, image-to-3D, unconditional or conditional 3D
 * generation via diffusion / autoregressive / GAN / VAE / feed-forward models.
 * Rejects pure 2D image generation, deterministic reconstruction/SLAM, novel-view
 * synthesis without generation, segmentation/classification, and non-3D work.
 * Also rejects HUMAN-centric generation (heads, faces, avatars, full bodies, hands,
 * hair) — this archive is general object/scene shape generation, not digital humans.
 */
const SYSTEM = `You tag 3D-shape-generation papers for a curated archive of GENERAL OBJECT & SCENE geometry.

Return STRICT JSON matching this exact schema:

{
  "is_shape_paper": boolean,         // true iff the paper's CENTRAL contribution is GENERATING general 3D object/scene shape/geometry (mesh, point cloud, voxel, SDF, implicit field, gaussian, triplane) via diffusion / autoregressive / GAN / VAE / flow / feed-forward / score-distillation. Includes text-to-3D, image-to-3D, unconditional & conditional 3D generation of objects/scenes/parts/CAD, and generative shape editing/completion. REJECT human-centric generation even though it is technically 3D: head/face/portrait avatars, full-body human avatars, hands, hair, talking heads, facial expression/animation, body/garment/clothing generation, animal/character avatars driven by humans. ALSO reject: pure 2D image/video generation with no 3D output, deterministic single-solution reconstruction or SLAM/MVS, novel-view synthesis (NeRF/3DGS) without a generative model, depth/normal estimation, segmentation/classification/retrieval, pose estimation, NLP, robotics.
  "reject_reason": string,          // empty if accepted; else short reason
  "short": string,                  // method short name, e.g. "Shap-E" or "MeshGPT". Kebab-case ok if no explicit name.
  "tags": string[],                 // 2-5 concise keywords. Pick from these axes when applicable:
                                    //   representation: {"Mesh","PointCloud","Voxel","SDF","Implicit","3DGS","Triplane","NeRF","Hybrid"}
                                    //   model: {"Diffusion","Autoregressive","GAN","VAE","Flow","LRM","Feed-forward","SDS"}
                                    //   conditioning: {"Text","Image","Single-image","Multi-view","Unconditional","Sketch","PointCloud"}
                                    //   capability: {"Generation","Completion","Editing","Texturing","Scene","Part-aware"}
  "contribution": string,           // one-line key novelty ("First X that does Y via Z")
  "summary": string,                // neutral paraphrased abstract, <= 400 chars
  "importance": 1|2|3|4|5,          // 1=minor, 3=solid venue quality, 5=field-defining SOTA
  "code_hint": boolean,             // true if the abstract mentions code/project page release
  "project_url_hint": string        // empty if unknown; else a URL if the abstract mentions one explicitly
}

Rules:
- Accept ONLY when the MAIN target is generating GENERAL object/scene 3D shape/geometry. A paper that reconstructs a single 3D model deterministically (no generative/probabilistic model, no sampling) is NOT shape generation — reject it.
- Reject digital-human work outright: if the primary subject is a human head, face, avatar, body, hand, or hair, set is_shape_paper=false even if the method is a 3D generative model. (A generic object generator that merely shows a head among many object categories is fine.)
- Tags array must have 2-5 items.
- Output ONLY the JSON object. No prose, no fences.`;

export async function tagPaper(paper: ArxivCandidate): Promise<TaggedFields | null> {
  const userBody = JSON.stringify({
    title: paper.title,
    authors: paper.authors,
    published: paper.published,
    abstract: paper.abstract,
    arxiv_id: paper.id,
    primary_category: paper.primaryCategory,
  });

  const resp = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 900,
    temperature: 0,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: userBody },
    ],
  });

  const text = resp.choices[0]?.message?.content ?? '';
  const json = extractJson(text);
  if (!json) {
    console.error('[tag] no JSON from model for', paper.id);
    return null;
  }

  return {
    is_shape_paper: Boolean(json.is_shape_paper),
    reject_reason: typeof json.reject_reason === 'string' ? json.reject_reason : undefined,
    short: String(json.short ?? paper.id),
    tags: Array.isArray(json.tags) ? json.tags.map(String).slice(0, 5) : [],
    contribution: String(json.contribution ?? ''),
    summary: String(json.summary ?? '').slice(0, 400),
    importance: clampImportance(json.importance),
    code_hint: Boolean(json.code_hint),
    project_url_hint: typeof json.project_url_hint === 'string' ? json.project_url_hint : undefined,
  };
}

function clampImportance(v: unknown): TaggedFields['importance'] {
  const n = Number(v);
  if (n >= 1 && n <= 5 && Number.isInteger(n)) return n as TaggedFields['importance'];
  return 2;
}

function extractJson(text: string): Record<string, unknown> | null {
  const cleaned = text
    .replace(/^```(?:json)?/i, '')
    .replace(/```\s*$/i, '')
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(cleaned.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}
