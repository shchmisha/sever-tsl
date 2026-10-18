export const meta = {
  name: 'label-leaf-grouping',
  description: 'Two independent passes per video: group sampled frames into visible sheets, type each sheet, choose reading frames',
  phases: [{ title: 'Group', detail: '2 independent labellers x 5 videos' }],
}
const ROOT = '/Users/shchm/Documents/север/analysis/labelling-20261018'
const VIDEOS = [
  { id: '20260821_144838', sheets: 5, frames: 102 },
  { id: '20260821_145110', sheets: 8, frames: 190 },
  { id: '20260818-6_131851', sheets: 7, frames: 154 },
  { id: '20260821_145342', sheets: 7, frames: 163 },
  { id: '20260818_131702', sheets: 5, frames: 106 },
]
const SCHEMA = { type: 'object', properties: {
  leaves: { type: 'array', items: { type: 'object', properties: {
    first_sample: { type: 'integer' }, last_sample: { type: 'integer' },
    document_type: { type: 'string', enum: ['upd', 'torg12', 'ttn', 'trn', 'unknown'] },
    observed_form: { type: 'string', description: 'Literal own heading/form as printed if readable (e.g. "Универсальный передаточный документ", "Товарная накладная ... ТОРГ-12", "Оборотная сторона формы № 1-Т", "Акт ..."), else a short description' },
    side: { type: 'string', enum: ['front', 'reverse', 'continuation', 'single', 'unclear'] },
    rotation_to_upright_clockwise: { type: 'integer', enum: [0, 90, 180, 270], description: 'Clockwise rotation that makes the printed text upright' },
    best_samples: { type: 'array', items: { type: 'integer' }, description: '2-4 sample ids where the sheet is sharp, fully visible and not occluded, spread over the showing' },
    repeat_of: { type: ['integer', 'null'], description: 'Index (0-based in this list) of an earlier leaf showing the same physical page again, else null' },
    occlusion_or_blur: { type: 'string' },
    notes: { type: 'string' } },
    required: ['first_sample', 'last_sample', 'document_type', 'observed_form', 'side', 'rotation_to_upright_clockwise', 'best_samples', 'repeat_of', 'occlusion_or_blur', 'notes'] } },
  transition_samples: { type: 'array', items: { type: 'integer' }, description: 'Samples during page turns that belong to no stable sheet' },
  confidence: { type: 'number' } }, required: ['leaves', 'transition_samples', 'confidence'] }

const RULES = `You are labelling a phone video of Russian shipping documents being shown one sheet after another, for an independent ground truth.
Inputs: contact sheets ${'${ROOT}'}/VIDEO/contact-sheet-NN.jpg (24 thumbnails each, captioned "#sample time") and full 720-px sampled frames ${'${ROOT}'}/VIDEO/sampled-frames/frame-NNN.jpg (NNN = sample id, 3 digits). Sample times are in ${'${ROOT}'}/VIDEO/sampling.json.
Look ONLY at these images and that JSON. Do not open any other file in the repository (no pipeline results, no traces, no existing ground truth).
Task: split the timeline into visible "leaves": maximal runs of samples showing the same physical sheet face steadily. A page turn ends a leaf; samples mid-turn go to transition_samples. If the same page is shown again later, it is a NEW leaf with repeat_of pointing at the earlier one. A brief steady showing of 1-2 samples is still a leaf.
Type each leaf by its own printed heading, opening the 720-px frames whenever a thumbnail is not enough:
- upd: "Универсальный передаточный документ" with the Статус box (1 or 2), invoice header "Счет-фактура №"; a УПД continuation page (table only, no header) is upd with side=continuation.
- ttn: товарно-транспортная накладная, form 1-Т; its reverse "Оборотная сторона формы № 1-Т" / "ТРАНСПОРТНЫЙ РАЗДЕЛ"; and its commodity-section continuation: a sheet headed "Товарная накладная" with "Страница 2" and/or the line "Товарная накладная имеет продолжение на ___ листах, на бланках за N ___" and no ТОРГ-12 code.
- torg12: "Товарная накладная" with the code/caption "ТОРГ-12" (ОКУД 0330212) and the line "имеет приложение на ___ листах и содержит ___ порядковых номеров записей".
- trn: "Транспортная накладная" (standalone, with Заказчик/Перевозчик sections).
- unknown: anything else (standalone счет-фактура without УПД status, CMR, ТН-2, acts, letters, declarations, certificates, waybills, blank backs). Give its literal heading in observed_form.
Choose best_samples for reading: sharp, the sheet's printed area fully in frame, no fingers over the header, spread over the showing.
Be exhaustive across ALL contact sheets of the video; every non-transition sample must belong to exactly one leaf. Return only the structured object.`

phase('Group')
const jobs = []
for (const v of VIDEOS) for (const pass of [1, 2]) jobs.push({ v, pass })
const results = await pipeline(jobs,
  j => agent(RULES.replaceAll('VIDEO', j.v.id).replaceAll('${ROOT}', ROOT) + `\n\nVideo: ${j.v.id} (${j.v.frames} samples, ${j.v.sheets} contact sheets). You are labeller #${j.pass}; another labeller works independently.`,
    { label: `group:${j.v.id}:${j.pass}`, phase: 'Group', schema: SCHEMA, model: 'opus' }),
  (r, j) => ({ video: j.v.id, pass: j.pass, result: r }))
return results.filter(Boolean)