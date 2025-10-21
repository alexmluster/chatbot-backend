// server.js (CommonJS)
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const OpenAI = require('openai');     // default export in CJS
const axios = require('axios');
const cheerio = require('cheerio');
const COMPLETIONS_MODEL = process.env.MODEL || 'gpt-4o-mini';
// conversational but grounded
const DEFAULT_TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.45);
const RESPONSE_TONE = process.env.RESPONSE_TONE || 'friendly, concise, professional';


dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// CORS: lock Netlify origins in prod via NETLIFY_ORIGINS env (comma-separated)
app.use(cors({
  origin: (process.env.NETLIFY_ORIGINS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .concat((process.env.NETLIFY_ORIGINS ? [] : [true]))[0]   // true if not set
}));
app.use(express.json({ limit: '1mb' }));

// ---------- OpenAI ----------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const COMPLETIONS_MODEL = process.env.MODEL || 'gpt-4o-mini';
const EMBEDDING_MODEL   = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';

// ---------- Basic user sessions (trimmed) ----------
const userSessions = new Map();
const MAX_TURNS = 8; // keep last N user/assistant pairs

function pushToSession(userId, role, content) {
  if (!userSessions.has(userId)) userSessions.set(userId, []);
  const arr = userSessions.get(userId);
  arr.push({ role, content: String(content) });
  while (arr.length > MAX_TURNS * 2) arr.splice(0, 2); // trim oldest
  return arr;
}

// ========== DOCS-ONLY RAG (the two Naviga manuals) ==========
// Hard whitelist (server-side enforcement)
const DOC_BASES = Object.freeze([
  'https://docs.navigaglobal.com/circulation-setup-manual',
  'https://docs.navigaglobal.com/circulation-user-manual',
]);

const MAX_PAGES      = Number(process.env.MAX_PAGES || 120);
const CHUNK_SIZE     = Number(process.env.CHUNK_SIZE || 1100);
const CHUNK_OVERLAP  = Number(process.env.CHUNK_OVERLAP || 160);

// In-memory index: [{ url, title, text, embedding }]
let RAG_INDEX = null;

function sameOriginPath(url, base) {
  try {
    const u = new URL(url, base);
    const b = new URL(base);
    return u.origin === b.origin && u.pathname.startsWith(b.pathname);
  } catch { return false; }
}
function isDocLike(href = '') {
  return href
    && !href.startsWith('#')
    && !href.match(/\.(png|jpe?g|gif|svg|webp|ico|css|js|pdf|zip|mp4|mp3|woff2?)($|\?)/i);
}
function extractMainText(html) {
  const $ = cheerio.load(html);
  const candidates = [
    'article','main','.content','.markdown','.md-content','.docs-content',
    'div[role=main]','.docMainContainer','.theme-doc-markdown','.page-content'
  ];
  let scope = $;
  for (const sel of candidates) {
    const n = $(sel);
    if (n.length) { scope = n; break; }
  }
  const parts = scope.find('h1,h2,h3,h4,h5,h6,p,li').map((_, el) => $(el).text().trim()).get();
  return parts.filter(Boolean).join('\n');
}
async function fetchPage(url) {
  const { data } = await axios.get(url, { timeout: 20000, headers: { 'User-Agent': 'NavigaDocsOnly/1.0' } });
  const $ = cheerio.load(data);
  const title = $('title').first().text().trim();
  const text = extractMainText(data);
  const links = $('a[href]').map((_, a) => $(a).attr('href')).get();
  return { url, title, text, links };
}
function chunkText(t, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const out = [];
  const s = String(t || '');
  if (!s.trim()) return out;
  for (let i = 0; i < s.length; i += (size - overlap)) {
    out.push(s.slice(i, Math.min(s.length, i + size)));
  }
  return out;
}
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

async function buildDocsIndex() {
  // Crawl within the two path prefixes (BFS, capped)
  const visited = new Set();
  const queue = [...DOC_BASES];
  const pages = [];

  while (queue.length && pages.length < MAX_PAGES) {
    const url = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);
    try {
      const p = await fetchPage(url);
      pages.push({ url: p.url, title: p.title, text: p.text });
      const base = DOC_BASES.find(b => sameOriginPath(url, b));
      if (base) {
        p.links
          .map(href => new URL(href, url).toString())
          .filter(h => isDocLike(h) && sameOriginPath(h, base))
          .forEach(h => {
            if (!visited.has(h) && pages.length + queue.length < MAX_PAGES) queue.push(h);
          });
      }
    } catch (e) {
      console.warn('[crawl] failed:', url, e.message);
    }
  }

  // Chunk + embed
  const records = [];
  for (const p of pages) {
    for (const c of chunkText(p.text)) {
      records.push({ url: p.url, title: p.title, text: c });
    }
  }

  // Embed in batches
  const BATCH = 100;
  for (let i = 0; i < records.length; i += BATCH) {
    const inputs = records.slice(i, i + BATCH).map(r => r.text);
    const emb = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: inputs });
    for (let j = 0; j < inputs.length; j++) {
      records[i + j].embedding = emb.data[j].embedding;
    }
  }
  RAG_INDEX = records;
  console.log('[docs-only] built index:', RAG_INDEX.length, 'chunks');
}

async function ensureIndex() {
  if (!RAG_INDEX || !RAG_INDEX.length) await buildDocsIndex();
}

async function retrieveDocs(query, k = 8) {
  await ensureIndex();
  const q = (await openai.embeddings.create({ model: EMBEDDING_MODEL, input: query })).data[0].embedding;
  return RAG_INDEX
    .map(r => ({ ...r, score: cosineSim(q, r.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .filter(r => r.score > 0.1);
}

async function answerFromDocs(question) {
  const top = await retrieveDocs(question, 8);
  if (!top.length) {
    return { reply: "I don’t see that in the Naviga Circulation manuals.", citations: [] };
  }
  const sources = top.map((r, i) => `# Source ${i+1} — ${r.title}\nURL: ${r.url}\n${r.text}`).join('\n\n---\n\n');
  const sys = [
    "You are a helpful assistant for Naviga Circulation operations.",
    "Answer ONLY using the 'Sources' text. If the sources don't answer it, say you don't see it in the manuals.",
    "Be concise and practical. Use short steps when helpful.",
    "End with a 'Source:' line citing 1–2 relevant URLs."
  ].join(' ');
  const user = `Question: ${question}\n\nSources:\n${sources}`;

  const resp = await openai.chat.completions.create({
    model: COMPLETIONS_MODEL,
    temperature: 0.2,
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: user }
    ]
  });

  const reply = resp.choices?.[0]?.message?.content?.trim()
    || "I don’t see that in the Naviga Circulation manuals.";
  const citations = top.slice(0, 2).map(r => ({ url: r.url, title: r.title }));
  return { reply, citations };
}

// ---------- Health ----------
app.get('/health', async (_req, res) => {
  res.json({ ok: true, indexed: RAG_INDEX?.length || 0 });
});

// ---------- Docs-only endpoint ----------
app.post('/docs-only', async (req, res) => {
  try {
    // Optional client-sent whitelist must be a subset; ignore anything else
    const allowedDocs = Array.isArray(req.body?.allowedDocs) ? req.body.allowedDocs : [];
    const invalid = allowedDocs.filter(u => !DOC_BASES.some(b => u && u.startsWith(b)));
    if (invalid.length) return res.status(403).json({ error: 'Only the two Naviga manuals are allowed.' });

    const question = String(req.body?.question || req.body?.message || '').trim();
    if (!question) return res.status(400).json({ error: 'Missing question' });

    const out = await answerFromDocs(question);
    res.json(out);
  } catch (e) {
    console.error('docs-only error:', e);
    res.status(500).json({ error: 'Server error' });
  }
});

// ---------- Chat endpoint (kept; routes to docs-only if requested) ----------
app.post('/chat', async (req, res) => {
  const { message, userId = 'default-user', mode, restrict } = req.body || {};
  try {
    if (!message) return res.status(400).json({ error: 'Message is required' });

    // If frontend is in kb-only mode, route to docs-only for guaranteed restriction.
    if (mode === 'kb-only' || restrict === 'kb+docs' || restrict === 'docs-only') {
      const out = await answerFromDocs(message);
      return res.json({ reply: out.reply, citations: out.citations, source: 'docs-only' });
    }

    // Otherwise: regular chat with short session history
    const session = pushToSession(userId, 'user', message);
    const completion = await openai.chat.completions.create({
      model: COMPLETIONS_MODEL,
      messages: session,
      temperature: 0.5
    });
    const reply = completion.choices[0]?.message?.content || 'Sorry, I didn’t quite catch that.';
    pushToSession(userId, 'assistant', reply);
    res.json({ reply });
  } catch (error) {
    console.error('OpenAI error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to fetch AI response' });
  }
});

// ---------- Root + Start ----------
app.get('/', (_req, res) => res.send('OK'));
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});


