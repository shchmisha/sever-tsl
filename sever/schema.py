from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal["upd", "torg12", "ttn", "trn", "unknown"]
FieldStatus = Literal["recognized", "not_present", "unreadable", "conflict"]
FIELDS_BY_TYPE: dict[str, list[str]] = {
    "upd": ["invoice_number", "invoice_date", "consignor", "consignor_address", "consignee", "consignee_address"],
    "torg12": ["supplier", "consignee", "document_number", "document_date"],
    "ttn": ["consignor", "consignee", "document_number", "document_date"],
    "trn": ["document_date", "document_number", "consignor", "customer", "consignee"],
    "unknown": [],
}

class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    region_index: int | None = None
    text: str
    bbox: list[float] | None = None  # x,y,width,height; normalized top-left; engine provenance defines the evidence image

class FieldValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_value: str | None = None
    normalized_value: str | None = None
    status: FieldStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def consistent_value(self):
        if self.status == "recognized" and not self.raw_value:
            raise ValueError("recognized fields require a raw value")
        if self.status in {"not_present", "unreadable", "conflict"} and self.normalized_value is not None:
            raise ValueError("unresolved fields must have null normalized_value")
        return self

class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    page_id: str
    document_type: DocumentType
    type_confidence: float | None = Field(default=None, ge=0, le=1)
    classification_evidence: list[Evidence] = Field(default_factory=list)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_contract(self):
        expected = set(FIELDS_BY_TYPE[self.document_type])
        if set(self.fields) != expected:
            raise ValueError(f"{self.document_type} requires exactly {sorted(expected)}")
        return self
