"""Source-bound printed INN metadata and optional public FNS verification.

This is separate from the 19 extracted fields. Registry names never overwrite
document names or shipping addresses. Local checksum validity is not proof that
an organization exists in EGRUL/EGRIP. The HTTP contract is the official public
site's observed frontend protocol, not a documented/stable integration API:
https://egrul.nalog.ru/js/index.js?ver=1.8.14&lang=RUS
https://egrul.nalog.ru/about.html
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import httpx

from .normalize import comparison
from .schema import DocumentResult, Evidence, FIELDS_BY_TYPE
from .vision import _box, _is_company, _organization_literal, _role_sources

FNS_ORIGIN = "https://egrul.nalog.ru"
FNS_VERSION = "public-egrul-search-v1"
_PARTIES = {"consignor", "consignee", "supplier", "customer"}
_INN_LABEL = re.compile(r"\bинн\b", re.I)
_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-я]{8,24}(?:\s*/\s*[0-9A-Za-zА-Яа-я]{8,12})?")
_LEGAL_NAME = re.compile(r'(?:Общество\s+с\s+ограниченной\s+ответственностью|Акционерное\s+общество|ООО|ПАО|ОАО|ЗАО|АО)\s+["«“]([^"»”]{1,100})["»”]', re.I)
_QUALIFIERS = {"seller": {"продавца"}, "consignor": {"грузоотправителя", "отправителя"},
               "consignee": {"грузополучателя", "получателя"}, "supplier": {"поставщика"}, "customer": {"заказчика"}}
_QUALIFIER = re.compile(r"^(продавца|покупателя|грузоотправителя|грузополучателя|получателя|отправителя|поставщика|заказчика|перевозчика|плательщика)\b", re.I)


def validate_inn(value: str | None) -> dict[str, Any]:
    """Check literal ASCII digits/length and standard control digits, no repair."""
    valid = isinstance(value, str) and bool(re.fullmatch(r"[0-9]{10}|[0-9]{12}", value))
    valid = valid and not value.startswith("00")
    checksum = None
    if valid:
        digits = [int(c) for c in value]
        if len(digits) == 10:
            checksum = sum(a*b for a,b in zip(digits, [2,4,10,3,5,9,4,6,8])) % 11 % 10 == digits[9]
        else:
            first = sum(a*b for a,b in zip(digits, [7,2,4,10,3,5,9,4,6,8])) % 11 % 10
            second = sum(a*b for a,b in zip(digits, [3,7,2,4,10,3,5,9,4,6,8])) % 11 % 10
            checksum = first == digits[10] and second == digits[11]
    return {"format_valid": bool(valid), "checksum_valid": checksum,
            "registration_verified": False,
            "method": "10/12 literal digits and weighted control digits; no registry evidence"}


def _evidence(page, indices):
    return [Evidence(frame_id=str(page["frame_id"]), region_index=i,
                     text=page["regions"][i].get("text", ""), bbox=_box(page["regions"][i])).model_dump()
            for i in dict.fromkeys(indices)]


def _candidate(text):
    # INN/KPP is a labeled pair. A literal slash separates them; never truncate
    # an unseparated 19/20-digit OCR run or replace O/I with digits.
    text = text.strip(" :;.,")
    match = _TOKEN.match(text)
    if not match:
        return None
    value = match.group().split("/")[0].strip()
    if not re.search(r"[0-9]", value):
        return None
    return value


def _role_inn(page, role, panels):
    panel = panels.get(role)
    refs = []
    resolved_role = role
    if role == "consignor" and panel and panel.get("explicit_seller_reference"):
        refs = [int(i[1:]) for i in panel.get("ids", []) if re.search(
            r"\bон\s+же\b|\bона\s+же\b", page["regions"][int(i[1:])].get("text", ""), re.I)]
        if refs:
            if panel.get("caption_id"):
                refs.insert(0, int(panel["caption_id"][1:]))
            panel, resolved_role = panels.get("seller"), "seller"
    if not panel:
        return {"printed_inn": None, "printed_status": "unavailable", "candidates": [], "evidence": [],
                "resolved_role": resolved_role, "company_name_binding": {"status": "unavailable", "legal_name_candidates": [], "evidence": []},
                "reason": "No bounded own party caption was legible in the selected-frame OCR"}
    ids = [int(identifier[1:]) for identifier in panel.get("ids", [])]
    regions = page.get("regions", [])
    found = []
    for i in ids:
        r = regions[i]; text = r.get("text", ""); b = _box(r)
        label = _INN_LABEL.search(text)
        if not label or not b:
            continue
        tail = text[label.end():]
        # A wrong-role qualifier is stronger than geometric proximity. A
        # cropped/misrecognized buyer column must not supply the consignee INN.
        tail = re.sub(r"^\s*(?:/\s*кпп)?\s*", "", tail, flags=re.I)
        qualifier = _QUALIFIER.match(tail)
        if qualifier:
            if qualifier.group().casefold() not in _QUALIFIERS.get(resolved_role, set()):
                continue
            tail = tail[qualifier.end():]
        tail = tail.lstrip(" :：")
        inline = _candidate(tail)
        if inline:
            found.append((inline, [i]))
            continue
        source_pass = r.get("processing_pass", "unknown")
        following = []
        for j in ids:
            value_region = regions[j]; box = _box(value_region)
            if not box or j == i or value_region.get("processing_pass", "unknown") != source_pass:
                continue
            line = value_region.get("text", "").strip()
            if not re.match(r"^[0-9A-Za-zА-Яа-я]{8,24}(?:\s*/|[,\s]|$)", line) or not re.match(r"^[0-9]", line):
                continue
            # Either a separate value alongside ИНН/КПП or the next wrapped
            # value line after a company line ending in ИНН. Role/column bounds
            # are already supplied by the actual same-frame party panel.
            inline_row = abs(box[1]-b[1]) <= .020 and box[0] >= b[0]+b[2]-.012
            wrapped = 0 <= box[1]-b[1] <= .045 and box[0] >= panel["bbox"][0]
            candidate = _candidate(line)
            if candidate and (inline_row or wrapped):
                following.append((abs(box[1]-b[1]), box[0], candidate, j))
        if following:
            _, _, value, j = min(following)
            found.append((value, [i, j]))
    valid_length = [(v, evidence) for v, evidence in found if re.fullmatch(r"[0-9]{10}|[0-9]{12}", v)]
    observations = valid_length or found
    values = sorted({value for value, _ in observations})
    caption_ids = [int(panel["caption_id"][1:])] if panel.get("caption_id") else []
    indices = [*caption_ids, *refs, *[i for _, proof in observations for i in proof]]
    binding_names, binding_rows = {}, []
    ambiguous_row = False
    for _, proof in observations:
        for i in proof:
            text = regions[i].get("text", "")
            if not _INN_LABEL.search(text):
                continue
            row_names = {comparison("consignor", match.group()): match.group() for match in _LEGAL_NAME.finditer(text)}
            if row_names:
                binding_names.update(row_names);binding_rows.append(i)
            ambiguous_row |= len(row_names) > 1
    binding = {"status": "ambiguous" if ambiguous_row else "unique" if len(binding_names) == 1 else "unavailable",
               "legal_name_candidates": list(binding_names.values()), "evidence": _evidence(page, [*caption_ids, *binding_rows]) if binding_rows else [],
               "reason": "Several literal legal names occur in the same own-role labeled-INN row; entity binding is unresolved" if ambiguous_row else
                         "Literal legal name alongside the own-role INN" if binding_names else "No legal name in the literal labeled-INN row"}
    return {"printed_inn": values[0] if len(values) == 1 else None,
            "printed_status": "found" if len(values) == 1 else "ambiguous" if values else "not_present",
            "candidates": values, "evidence": _evidence(page, indices), "resolved_role": resolved_role, "company_name_binding": binding,
            "reason": "Literal labeled INN in own-role selected-frame OCR; no digit repair" if values else
                      "No legible labeled INN observed in the bounded role OCR; this does not prove printed absence"}


def _literal_names(page, panel):
    names = {}
    if panel:
        for identifier in panel.get("ids", []):
            text = page["regions"][int(identifier[1:])].get("text", "")
            literal = _organization_literal(text) if _is_company(text) else None
            if literal:
                names.setdefault(comparison("consignor", literal), literal)
    return names


def extract_printed_inns(ocr_pages: list[dict], document: DocumentResult | dict) -> dict:
    """Read party INNs from field source frames; incompatible views are ambiguous.

    Field evidence restricts eligible source views. Without it, inspect all
    supplied leaf views without majority/checksum voting. An unavailable field
    source cannot be replaced with another frame. «Он же» stays in each frame.
    """
    document = document.model_dump() if isinstance(document, DocumentResult) else document
    kind = document["document_type"]
    result = {"source_id": document["source_id"], "page_id": document["page_id"], "roles": {},
              "method": "field-source-frame labeled OCR inside own party bands; same-frame seller resolution only; no majority vote",
              "limitations": ["Checksum is not registration proof", "Registry results never replace printed names or shipping addresses"]}
    if not ocr_pages or kind == "unknown":
        return result
    for role in FIELDS_BY_TYPE[kind]:
        if role not in _PARTIES:
            continue
        field = document.get("fields", {}).get(role, {})
        address = document.get("fields", {}).get(role + "_address", {})
        frame_ids = {proof.get("frame_id") for f in (field, address) for proof in f.get("evidence", []) if proof.get("frame_id")}
        available = {str(page["frame_id"]) for page in ocr_pages}
        selected = [page for page in ocr_pages if not frame_ids or str(page["frame_id"]) in frame_ids]
        if frame_ids - available:
            selected = []
        observations, names = [], {}
        for page in selected:
            panels = _role_sources(page, kind)
            observed = _role_inn(page, role, panels)
            observed.update(frame_id=str(page["frame_id"]), timestamp_s=page.get("timestamp_s"))
            observations.append(observed)
            names.update(_literal_names(page, panels.get(observed["resolved_role"])))
        all_values = {value for observed in observations for value in observed["candidates"]}
        digit_values = {value for value in all_values if re.fullmatch(r"[0-9]{10}|[0-9]{12}", value)}
        values = sorted(digit_values or all_values)
        status = "found" if len(values) == 1 else "ambiguous" if values else (
            "not_present" if any(o["printed_status"] == "not_present" for o in observations) else "unavailable")
        reading = {"printed_inn": values[0] if len(values) == 1 else None,
                   "printed_status": status, "candidates": values,
                   "evidence": [proof for observed in observations for proof in observed["evidence"]],
                   "source_frame_ids": list(dict.fromkeys(o["frame_id"] for o in observations)),
                   "frame_observations": observations,
                   "resolved_role": observations[0]["resolved_role"] if observations and len({o["resolved_role"] for o in observations}) == 1 else None,
                   "reason": "Conflicting literal own-role INNs; no majority/checksum selection" if status == "ambiguous" else
                             "Field evidence refers to a source frame not provided" if frame_ids - available else
                             "Literal labeled own-role OCR; malformed alternatives retained in frame_observations"}
        reading["local_validation"] = validate_inn(reading["printed_inn"])
        name = ((field.get("normalized_value") or field.get("raw_value"))
                if frame_ids and selected and field.get("status") == "recognized" else None)
        reading["name_source"] = "extracted_field_evidence" if name else "same_role_ocr_literal" if len(names) == 1 else "unavailable"
        # No registry-derived repair. Without field frame evidence the name
        # must be a unique literal own-role company across supplied source views.
        reading["printed_company_name"] = name or (next(iter(names.values())) if len(names) == 1 else None)
        row_bindings = [observed["company_name_binding"] for observed in observations]
        candidates = {comparison("consignor", literal): literal for binding in row_bindings for literal in binding["legal_name_candidates"]}
        ambiguous = any(binding["status"] == "ambiguous" for binding in row_bindings)
        reading["company_name_binding"] = {
            "status": "ambiguous" if ambiguous else "unique" if reading["printed_company_name"] else "unavailable",
            "legal_name_candidates": list(candidates.values()) or ([reading["printed_company_name"]] if reading["printed_company_name"] else []),
            "evidence": [proof for binding in row_bindings for proof in binding["evidence"]],
            "extracted_company_name": reading["printed_company_name"],
            "reason": "Multiple literal legal names share an actual own-role labeled-INN row; keep the extracted field and INN, but do not declare a unique registry name match" if ambiguous else
                      "Company name from field evidence or unique same-role literal OCR; no registry substitution"}
        result["roles"][role] = reading
    return result


def _unavailable(reason, *, failure_class=None, checked_at=None):
    return {"status": "unavailable", "reason": reason, "failure_class": failure_class,
            "checked_at": checked_at, "source_url": FNS_ORIGIN + "/", "records": []}


class FnsRegistry:
    """Bounded cached access to the official public EGRUL/EGRIP search.

    No key, paid intermediary, CAPTCHA solving, PDF requests, automatic retry or
    arbitrary provider URL. Each unique lookup makes at most four HTTP requests.
    One instance is the budget boundary and should be shared across job pages.
    """
    def __init__(self, cache_dir: Path, *, max_unique_lookups: int = 10,
                 timeout_seconds: float = 10, cache_ttl_seconds: float = 86400,
                 failure_ttl_seconds: float = 300, transport: httpx.AsyncBaseTransport | None = None):
        if not 0 <= max_unique_lookups <= 10:
            raise ValueError("The initial registry budget must be between 0 and 10 unique lookups")
        self.cache_dir = Path(cache_dir).resolve(); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_unique_lookups = max_unique_lookups
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.failure_ttl_seconds = failure_ttl_seconds
        self.transport = transport
        self.unique_lookups = 0; self.http_requests = 0; self.cache_hits = 0
        self._memo = {}; self._lock = asyncio.Lock()

    def _path(self, inn):
        digest = hashlib.sha256((FNS_VERSION + "|" + inn).encode()).hexdigest()
        return self.cache_dir / f"fns-{digest}.json"

    async def _request(self, client, method, path, **kwargs):
        self.http_requests += 1
        response = await client.request(method, FNS_ORIGIN + path, **kwargs)
        response.raise_for_status()
        if response.is_redirect:
            raise _RegistryFailure("unexpected_redirect")
        if len(response.content) > 2_000_000:
            raise _RegistryFailure("oversized_response")
        return response

    async def _fetch(self, inn):
        checked_at = datetime.now(timezone.utc).isoformat()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport,
                                         follow_redirects=False, headers={"User-Agent": "Sever-INN-verification/0.1", "Accept": "application/json,text/html"}) as client:
                home = await self._request(client, "GET", "/index.html")
                if re.search(r"CAPTCHA_REQUIRED\s*=\s*true", home.text, re.I):
                    raise _RegistryFailure("captcha_required")
                query = await self._request(client, "POST", "/", data={"query": inn, "region": "", "page": "", "vyp3CaptchaToken": ""},
                                            headers={"X-Requested-With": "XMLHttpRequest", "Referer": FNS_ORIGIN + "/index.html"})
                data = query.json()
                if not isinstance(data, dict):
                    raise _RegistryFailure("invalid_response")
                errors = data.get("ERRORS", data.get("errors", {}))
                if isinstance(errors, dict) and any("captcha" in key.casefold() for key in errors):
                    raise _RegistryFailure("captcha_required")
                token = data.get("t")
                if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", token):
                    raise _RegistryFailure("captcha_required" if data.get("captchaRequired") else "invalid_response")
                for poll in range(2):
                    response = await self._request(client, "GET", "/search-result/" + token, headers={"X-Requested-With": "XMLHttpRequest"})
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise _RegistryFailure("invalid_response")
                    if payload.get("status") == "wait":
                        if poll == 0:
                            await asyncio.sleep(.25)
                            continue
                        raise _RegistryFailure("result_pending")
                    if payload.get("captchaRequired"):
                        raise _RegistryFailure("captcha_required")
                    rows = payload.get("rows")
                    if not isinstance(rows, list):
                        raise _RegistryFailure("invalid_response")
                    # Public record fields only. Never save tokens, cookies,
                    # authentication/captcha data, HTML or registry addresses.
                    records = [{"inn": row.get("i"), "name": row.get("n"), "short_name": row.get("c"),
                                "ogrn": row.get("o"), "entity_type": row.get("k"),
                                "registration_date": row.get("r"), "termination_date": row.get("e"),
                                "invalid_registration_date": row.get("v")}
                               for row in rows if isinstance(row, dict) and row.get("k") in {"ul", "fl"}]
                    malformed_rows = any(not isinstance(row, dict) or
                                         row.get("k") not in {"ul", "fl"} and (row.get("i") or row.get("n") or str(row.get("cnt", "0")) != "0")
                                         for row in rows)
                    if (malformed_rows or len(records) > 20 or
                            any(not isinstance(r["inn"], str) or not isinstance(r["name"], str) or
                                any(value is not None and (not isinstance(value, str) or len(value) > 2000)
                                    for value in r.values()) for r in records)):
                        raise _RegistryFailure("invalid_response")
                    return {"status": "found" if records else "not_present", "checked_at": checked_at,
                            "source_url": FNS_ORIGIN + "/", "records": records,
                            "reason": "Official public EGRUL/EGRIP search response; not a signed extract"}
        except _RegistryFailure as exc:
            return _unavailable("Official public FNS service did not provide a usable registry response", failure_class=exc.code, checked_at=checked_at)
        except Exception as exc:
            # A public-service failure is metadata, not a failed document job.
            # Exception text can include session tokens; retain only its class.
            return _unavailable("Official public FNS request failed", failure_class=type(exc).__name__, checked_at=checked_at)

    async def lookup(self, inn: str, printed_company_name: str | None) -> dict:
        local = validate_inn(inn)
        if not local["format_valid"] or not local["checksum_valid"]:
            return _unavailable("Invalid literal INN format/checksum; registry was not queried", failure_class="invalid_inn")
        async with self._lock:
            cached = False
            raw = self._memo.get(inn)
            path = self._path(inn)
            if raw is not None:
                cached = True
            if raw is None and path.is_file():
                try:
                    saved = json.loads(path.read_text())
                    ttl = self.failure_ttl_seconds if saved["result"]["status"] == "unavailable" else self.cache_ttl_seconds
                    if (saved.get("version") == FNS_VERSION and saved.get("inn") == inn and saved.get("origin") == FNS_ORIGIN
                            and 0 <= time.time()-saved["created_at"] <= ttl):
                        raw = saved["result"]; cached = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if raw is None:
                if self.unique_lookups >= self.max_unique_lookups:
                    return _unavailable("Shared unique official lookup budget exhausted; no request made", failure_class="lookup_budget")
                self.unique_lookups += 1
                raw = await self._fetch(inn)
                # Atomic local file, containing only the allowlisted public result.
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"version": FNS_VERSION, "origin": FNS_ORIGIN, "inn": inn,
                    "created_at": time.time(), "result": raw}, ensure_ascii=False, indent=2))
                temporary.replace(path)
            if cached:
                self.cache_hits += 1
            self._memo[inn] = raw
        result = deepcopy(raw); result["cache_hit"] = cached
        if result["status"] != "found":
            return result
        records = [record for record in result["records"] if record["inn"] == inn]
        if not records:
            result.update(status="mismatch", reason="Official result INN differs from the printed INN")
        elif len(records) != 1:
            result.update(status="ambiguous", reason="Multiple official records have the exact printed INN")
        elif not printed_company_name:
            result.update(status="unavailable", failure_class="printed_name_unreadable",
                          reason="Official INN record found but printed company name is unreadable; name match cannot be verified")
        else:
            aliases = [records[0].get("name"), records[0].get("short_name")]
            printed = comparison("consignor", printed_company_name)
            if records[0].get("entity_type") == "fl":
                # EGRIP returns the exact person's name without an ИП prefix.
                printed = re.sub(r"^ип\s+", "", printed)
            matched = any(comparison("consignor", name) == printed for name in aliases if name)
            result.update(status="matched" if matched else "mismatch",
                          reason="Printed INN and legal entity name match the official public record" if matched else
                                 "Printed INN exists, but the printed company name differs from the official record",
                          name_comparison="declared legal-form, quote and punctuation normalization; no fuzzy name repair")
        return result

    def stats(self):
        return {"unique_lookups": self.unique_lookups, "max_unique_lookups": self.max_unique_lookups,
                "http_requests": self.http_requests, "cache_hits": self.cache_hits}


class _RegistryFailure(Exception):
    def __init__(self, code):
        self.code = code


async def verify_document_inns(ocr_pages: list[dict], document: DocumentResult | dict, *, registry: FnsRegistry | None = None) -> dict:
    result = extract_printed_inns(ocr_pages, document)
    for reading in result["roles"].values():
        if reading["printed_status"] == "ambiguous":
            reading["registry"] = {**_unavailable("Printed INN observations conflict; registry was not queried"), "status": "ambiguous"}
        elif reading["printed_status"] != "found":
            reading["registry"] = _unavailable("No uniquely legible printed INN in the bounded role; registry was not queried", failure_class="printed_inn_unavailable")
        elif registry is None:
            reading["registry"] = _unavailable("Optional official registry lookup disabled; local checksum is not registration proof", failure_class="lookup_disabled")
        else:
            ambiguous = reading["company_name_binding"]["status"] == "ambiguous"
            observed = await registry.lookup(reading["printed_inn"], None if ambiguous else reading["printed_company_name"])
            if ambiguous and observed.get("records") and observed.get("failure_class") == "printed_name_unreadable":
                observed.update(status="ambiguous", reason="Official INN record retained; several literal legal names share the document's own-role INN row, so company-name binding cannot be verified")
                observed.pop("failure_class", None)
            reading["registry"] = observed
    return result
