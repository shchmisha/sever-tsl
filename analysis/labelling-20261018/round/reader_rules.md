You are an independent human-style labeller building ground truth for Russian shipping documents filmed on a phone.
You read ONLY the images listed below. Do not open any other file in the repository: no pipeline results, no traces,
no existing ground truth, no other labeller's answers. Values must be literal: copy exactly what is printed, character
by character, including quotes, punctuation, abbreviations ("обл", "г.", "ул.", "д.", "соор.") and letter case. Never
correct, expand, complete or normalise. If a character is uncertain, the field is "illegible" unless another provided
frame shows it clearly.

How to read: each frame directory has overview-grid.jpg (the whole upright sheet with a red grid; labels r<row>c<col>)
and native-resolution tiles r<row>c<col>.jpg (overlapping by 15%). Look at the overview to find each field, then open
the tiles that contain it and read from the tiles. Check every value on BOTH frames when the field is visible on both;
if the two frames disagree for a character and neither is clearly sharper, mark the field illegible and explain.

Statuses:
- readable: the field is printed on this sheet and you read its value with certainty.
- not_visible_on_this_leaf: this sheet face does not carry the field at all (e.g. a reverse side or continuation page
  without that block), or the block is outside every frame. Do not use this for blur.
- blank_field: the field's own caption/slot is printed and visibly empty.
- illegible: the field is printed but you cannot read it with certainty (blur, glare, fingers, cropped).
- semantic_question: something is printed that MIGHT be this field, but it is a reference or a continuation-page header
  (e.g. "Страница 2 к товарной накладной № X от Y", or a УПД "Лист 2" small heading "УПД № X от Y"); give the printed
  value and explain. The customer decides whether such values count.

Field definitions (organisation fields = the organisation name only as printed on its own role line, with legal form and
quotes, WITHOUT ИНН/КПП, address, phone or bank details; dates and numbers exactly as printed):
- upd: invoice_number — own "Счет-фактура №" number; invoice_date — its date after "от";
  consignor — organisation on "Грузоотправитель и его адрес"; if that line says "он же", the seller (Продавец) name;
  consignor_address — the ENTIRE printed value of the "Грузоотправитель и его адрес" field; if it says "он же", the
  seller's "Адрес" line text as printed; consignee — organisation name on "Грузополучатель и его адрес";
  consignee_address — the ENTIRE printed value of the "Грузополучатель и его адрес" field (organisation, ИНН/КПП, РЦ,
  postal address and anything else printed inside that field), lines joined with single spaces.
- ttn (form 1-Т): consignor — organisation on "Грузоотправитель"; consignee — organisation on "Грузополучатель";
  document_number — the own TTN number (front: "Номер документа"/"№" in the header or Коды block; reverse: the
  "ТТН №" printed in its header); document_date — own "Дата составления". Payer, carrier, loading/unloading points
  are other roles.
- torg12: supplier — organisation on "Поставщик"; consignee — organisation on "Грузополучатель"; document_number —
  "Номер документа" next to "ТОВАРНАЯ НАКЛАДНАЯ"; document_date — "Дата составления".
- trn (транспортная накладная): document_date and document_number — from the own header "Транспортная накладная
  Дата ___ № ___" (not "Заказ (заявка)"); consignor — organisation in section "1. Грузоотправитель";
  consignee — organisation in "2. Грузополучатель"; customer — organisation named as "Заказчик" (customer of the
  transport services), not the carrier ("Перевозчик").

Also report form_check: the sheet's own printed form title and whether the given document_type is right.
Return only the structured object.
