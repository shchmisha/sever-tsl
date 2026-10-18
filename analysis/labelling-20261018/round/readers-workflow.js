export const meta = {
  name: 'label-field-readers',
  description: 'Two independent readers per target sheet transcribe every required field from native tiles; a third reader settles disagreements',
  phases: [{ title: 'Read', detail: '2 readers per sheet' }, { title: 'Settle', detail: 'third reader on disputed fields' }],
}

const RULES = args.rules
const FIELDS = { upd: ['invoice_number', 'invoice_date', 'consignor', 'consignor_address', 'consignee', 'consignee_address'],
  ttn: ['consignor', 'consignee', 'document_number', 'document_date'],
  torg12: ['supplier', 'consignee', 'document_number', 'document_date'],
  trn: ['document_date', 'document_number', 'consignor', 'customer', 'consignee'] }
const SCHEMA = { type: 'object', properties: {
  form_check: { type: 'object', properties: { printed_title: { type: 'string' }, type_correct: { type: 'boolean' }, note: { type: 'string' } },
    required: ['printed_title', 'type_correct', 'note'] },
  fields: { type: 'array', items: { type: 'object', properties: {
    name: { type: 'string' },
    status: { type: 'string', enum: ['readable', 'not_visible_on_this_leaf', 'blank_field', 'illegible', 'semantic_question'] },
    value: { type: ['string', 'null'], description: 'Literal printed value for readable/semantic_question, else null' },
    frames_used: { type: 'array', items: { type: 'integer' } },
    tiles: { type: 'array', items: { type: 'string' } },
    note: { type: 'string' } }, required: ['name', 'status', 'value', 'frames_used', 'tiles', 'note'] } } },
  required: ['form_check', 'fields'] }

const norm = v => (v == null ? null : v.replace(/\s+/g, ' ').trim())
const key = f => `${f.status}|${norm(f.value)}`

function prompt(leaf, names, reader) {
  const frames = leaf.frames.map(f => `- sample ${f.sample_id}: ${f.dir}`).join('\n')
  return `${RULES}\n\nSheet: ${leaf.video} ${leaf.leaf}, document_type given by the grouping labellers: ${leaf.document_type} (${leaf.side}); printed form noted: ${leaf.observed_form}\nFrames (each directory has overview-grid.jpg and tiles r<row>c<col>.jpg):\n${frames}\nReport exactly these fields, one entry each: ${names.join(', ')}.\nYou are reader #${reader}; others read independently.`
}

phase('Read')
const results = await pipeline(args.leaves,
  async leaf => {
    const names = FIELDS[leaf.document_type]
    const pair = await parallel([1, 2].map(n => () =>
      agent(prompt(leaf, names, n), { label: `read:${leaf.video.slice(-6)}:${leaf.leaf}:${n}`, phase: 'Read', schema: SCHEMA, model: 'opus' })))
    return { leaf, names, readers: pair }
  },
  async ({ leaf, names, readers }) => {
    const [a, b] = readers
    const byName = r => Object.fromEntries(((r && r.fields) || []).map(f => [f.name, f]))
    const fa = byName(a), fb = byName(b)
    const disputed = names.filter(n => !fa[n] || !fb[n] || key(fa[n]) !== key(fb[n]))
    let third = null
    if (disputed.length) {
      third = await agent(prompt(leaf, disputed, 3), { label: `settle:${leaf.video.slice(-6)}:${leaf.leaf}`, phase: 'Settle', schema: SCHEMA, model: 'opus', effort: 'high' })
    }
    const fc = byName(third)
    const fields = {}
    for (const n of names) {
      const cands = [fa[n], fb[n], fc[n]].filter(Boolean)
      if (fa[n] && fb[n] && key(fa[n]) === key(fb[n])) { fields[n] = { ...fa[n], agreement: '2_of_2', readings: [fa[n], fb[n]] }; continue }
      const counts = {}
      for (const c of cands) counts[key(c)] = (counts[key(c)] || 0) + 1
      const best = Object.entries(counts).sort((x, y) => y[1] - x[1])[0]
      if (best && best[1] >= 2) {
        const pick = cands.find(c => key(c) === best[0])
        fields[n] = { ...pick, agreement: '2_of_3', readings: cands }
      } else {
        fields[n] = { name: n, status: 'unresolved', value: null, agreement: 'none', readings: cands }
      }
    }
    return { video: leaf.video, leaf: leaf.leaf, document_type: leaf.document_type, form_checks: [a, b, third].filter(Boolean).map(r => r.form_check), fields }
  })
return results.filter(Boolean)
