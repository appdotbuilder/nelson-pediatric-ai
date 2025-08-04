from sqlmodel import SQLModel, Field, Relationship, JSON, Column, Text
from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum


# Enums for categorical data
class UserRole(str, Enum):
    STUDENT = "student"
    RESIDENT = "resident"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EmergencyType(str, Enum):
    NEONATAL_RESUSCITATION = "neonatal_resuscitation"
    ANAPHYLAXIS = "anaphylaxis"
    CARDIAC_ARREST = "cardiac_arrest"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    SEIZURES = "seizures"
    SHOCK = "shock"
    POISONING = "poisoning"


class DevelopmentalDomain(str, Enum):
    GROSS_MOTOR = "gross_motor"
    FINE_MOTOR = "fine_motor"
    LANGUAGE = "language"
    COGNITIVE = "cognitive"
    SOCIAL_EMOTIONAL = "social_emotional"
    ADAPTIVE = "adaptive"


class DrugUnit(str, Enum):
    MG_KG = "mg/kg"
    MCG_KG = "mcg/kg"
    UNITS_KG = "units/kg"
    ML_KG = "ml/kg"
    MG_KG_DAY = "mg/kg/day"
    MCG_KG_MIN = "mcg/kg/min"


# User Management Models
class User(SQLModel, table=True):
    __tablename__ = "users"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, max_length=50)
    email: str = Field(unique=True, max_length=255, regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    full_name: str = Field(max_length=100)
    role: UserRole = Field(default=UserRole.STUDENT)
    institution: Optional[str] = Field(default=None, max_length=200)
    specialty: Optional[str] = Field(default=None, max_length=100)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(default=None)
    preferences: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Relationships
    chat_sessions: List["ChatSession"] = Relationship(back_populates="user")


# Chat System Models
class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    title: str = Field(max_length=200)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_archived: bool = Field(default=False)
    session_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Relationships
    user: User = Relationship(back_populates="chat_sessions")
    messages: List["Message"] = Relationship(back_populates="chat_session", cascade_delete=True)


class Message(SQLModel, table=True):
    __tablename__ = "messages"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    chat_session_id: int = Field(foreign_key="chat_sessions.id")
    role: MessageRole
    content: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    token_count: Optional[int] = Field(default=None)
    processing_time: Optional[Decimal] = Field(default=None, decimal_places=3)

    # For assistant messages with citations
    citations: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    source_chunks: List[str] = Field(default=[], sa_column=Column(JSON))

    # Message metadata (tools used, confidence scores, etc.)
    message_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Relationships
    chat_session: ChatSession = Relationship(back_populates="messages")


# Medical Knowledge Base Models
class NelsonChunk(SQLModel, table=True):
    __tablename__ = "nelson_chunks"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="nelson_chapters.id")
    content: str = Field(sa_column=Column(Text))
    chunk_index: int = Field(default=0)
    token_count: int = Field(default=0)

    # Vector embedding for RAG (stored as JSON array for pgvector compatibility)
    embedding: List[float] = Field(default=[], sa_column=Column(JSON))

    # Metadata for chunk identification
    page_numbers: List[int] = Field(default=[], sa_column=Column(JSON))
    section_title: Optional[str] = Field(default=None, max_length=500)
    subsection_title: Optional[str] = Field(default=None, max_length=500)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    chapter: "NelsonChapter" = Relationship(back_populates="chunks")


class NelsonChapter(SQLModel, table=True):
    __tablename__ = "nelson_chapters"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_number: int
    title: str = Field(max_length=500)
    authors: List[str] = Field(default=[], sa_column=Column(JSON))
    edition: str = Field(max_length=20, default="22nd")
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    keywords: List[str] = Field(default=[], sa_column=Column(JSON))
    summary: Optional[str] = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    chunks: List[NelsonChunk] = Relationship(back_populates="chapter", cascade_delete=True)


# Pediatric Drug Dosage Models
class PediatricDrug(SQLModel, table=True):
    __tablename__ = "pediatric_drugs"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    generic_name: str = Field(max_length=200)
    brand_names: List[str] = Field(default=[], sa_column=Column(JSON))
    drug_class: str = Field(max_length=100)
    indications: List[str] = Field(default=[], sa_column=Column(JSON))
    contraindications: List[str] = Field(default=[], sa_column=Column(JSON))
    warnings: List[str] = Field(default=[], sa_column=Column(JSON))

    # Age/weight restrictions
    min_age_months: Optional[int] = Field(default=None)
    max_age_months: Optional[int] = Field(default=None)
    min_weight_kg: Optional[Decimal] = Field(default=None, decimal_places=2)
    max_weight_kg: Optional[Decimal] = Field(default=None, decimal_places=2)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    dosages: List["DrugDosage"] = Relationship(back_populates="drug", cascade_delete=True)


class DrugDosage(SQLModel, table=True):
    __tablename__ = "drug_dosages"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    drug_id: int = Field(foreign_key="pediatric_drugs.id")
    indication: str = Field(max_length=200)
    route: str = Field(max_length=50)  # oral, IV, IM, etc.

    # Dosage information
    dose_amount: Decimal = Field(decimal_places=3)
    dose_unit: DrugUnit
    frequency: str = Field(max_length=50)  # "q6h", "BID", "PRN", etc.
    max_daily_dose: Optional[Decimal] = Field(default=None, decimal_places=3)
    max_single_dose: Optional[Decimal] = Field(default=None, decimal_places=3)

    # Age/weight specific dosing
    min_age_months: Optional[int] = Field(default=None)
    max_age_months: Optional[int] = Field(default=None)
    min_weight_kg: Optional[Decimal] = Field(default=None, decimal_places=2)
    max_weight_kg: Optional[Decimal] = Field(default=None, decimal_places=2)

    # Additional instructions
    administration_notes: Optional[str] = Field(default=None, sa_column=Column(Text))
    monitoring_requirements: List[str] = Field(default=[], sa_column=Column(JSON))

    # Relationships
    drug: PediatricDrug = Relationship(back_populates="dosages")


# Emergency Protocol Models
class EmergencyProtocol(SQLModel, table=True):
    __tablename__ = "emergency_protocols"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200)
    protocol_type: EmergencyType
    age_group: str = Field(max_length=50)  # "neonate", "infant", "child", "adolescent", "all"
    keywords: List[str] = Field(default=[], sa_column=Column(JSON))

    # Protocol content
    overview: str = Field(sa_column=Column(Text))
    steps: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))  # ordered steps with actions
    medications: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))  # emergency meds and doses
    equipment: List[str] = Field(default=[], sa_column=Column(JSON))

    # Clinical decision support
    warning_signs: List[str] = Field(default=[], sa_column=Column(JSON))
    contraindications: List[str] = Field(default=[], sa_column=Column(JSON))
    when_to_call_help: List[str] = Field(default=[], sa_column=Column(JSON))

    # Metadata
    priority_level: int = Field(default=1)  # 1 = highest priority
    last_reviewed: Optional[datetime] = Field(default=None)
    source_references: List[str] = Field(default=[], sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Developmental Milestone Models
class DevelopmentalMilestone(SQLModel, table=True):
    __tablename__ = "developmental_milestones"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    age_months: int
    domain: DevelopmentalDomain
    milestone_text: str = Field(max_length=500)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))

    # Milestone timing
    typical_age_range_start: int  # months
    typical_age_range_end: int  # months
    red_flag_age: Optional[int] = Field(default=None)  # age when absence is concerning

    # Assessment details
    assessment_method: Optional[str] = Field(default=None, max_length=200)
    parent_report_acceptable: bool = Field(default=True)
    requires_observation: bool = Field(default=False)

    # References and notes
    source_references: List[str] = Field(default=[], sa_column=Column(JSON))
    clinical_notes: Optional[str] = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(default_factory=datetime.utcnow)


# Growth Chart Models
class GrowthChart(SQLModel, table=True):
    __tablename__ = "growth_charts"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    chart_type: str = Field(max_length=50)  # "weight-for-age", "height-for-age", "BMI-for-age", etc.
    sex: str = Field(max_length=10)  # "male", "female", "all"
    age_range_start: int = Field(default=0)  # age in months
    age_range_end: int = Field(default=240)  # age in months (20 years)

    # Chart data (percentile curves)
    percentile_data: Dict[str, List[Dict[str, float]]] = Field(default={}, sa_column=Column(JSON))
    # Structure: {"P3": [{"age": 0, "value": 2.5}, ...], "P50": [...], "P97": [...]}

    # Metadata
    source: str = Field(max_length=100, default="WHO/CDC")
    version: str = Field(max_length=20, default="2000")
    last_updated: Optional[datetime] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)


# Symptom and Clinical Decision Support Models
class Symptom(SQLModel, table=True):
    __tablename__ = "symptoms"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200)
    synonyms: List[str] = Field(default=[], sa_column=Column(JSON))
    category: str = Field(max_length=100)  # "respiratory", "gastrointestinal", etc.
    description: Optional[str] = Field(default=None, sa_column=Column(Text))

    # Age-specific information
    common_age_groups: List[str] = Field(default=[], sa_column=Column(JSON))
    red_flags: List[str] = Field(default=[], sa_column=Column(JSON))

    # Associated conditions
    common_diagnoses: List[str] = Field(default=[], sa_column=Column(JSON))
    urgent_diagnoses: List[str] = Field(default=[], sa_column=Column(JSON))

    # Clinical assessment
    assessment_questions: List[str] = Field(default=[], sa_column=Column(JSON))
    physical_exam_findings: List[str] = Field(default=[], sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Query and Search History Models
class SearchQuery(SQLModel, table=True):
    __tablename__ = "search_queries"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    query_text: str = Field(max_length=1000)
    query_type: str = Field(max_length=50)  # "chat", "drug_lookup", "emergency", "milestone", "symptom"
    results_count: int = Field(default=0)

    # Query context and metadata
    context_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    response_time: Optional[Decimal] = Field(default=None, decimal_places=3)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    user: Optional[User] = Relationship()


# Non-persistent schemas for API and validation
class UserCreate(SQLModel, table=False):
    username: str = Field(max_length=50)
    email: str = Field(max_length=255)
    full_name: str = Field(max_length=100)
    role: UserRole = Field(default=UserRole.STUDENT)
    institution: Optional[str] = Field(default=None, max_length=200)
    specialty: Optional[str] = Field(default=None, max_length=100)


class UserUpdate(SQLModel, table=False):
    full_name: Optional[str] = Field(default=None, max_length=100)
    institution: Optional[str] = Field(default=None, max_length=200)
    specialty: Optional[str] = Field(default=None, max_length=100)
    preferences: Optional[Dict[str, Any]] = Field(default=None)


class ChatSessionCreate(SQLModel, table=False):
    title: str = Field(max_length=200, default="New Chat")


class MessageCreate(SQLModel, table=False):
    role: MessageRole
    content: str
    citations: Optional[List[Dict[str, Any]]] = Field(default=None)
    message_metadata: Optional[Dict[str, Any]] = Field(default=None)


class DrugDosageQuery(SQLModel, table=False):
    drug_name: str = Field(max_length=200)
    weight_kg: Decimal = Field(decimal_places=2)
    age_months: Optional[int] = Field(default=None)
    indication: Optional[str] = Field(default=None, max_length=200)


class EmergencyProtocolQuery(SQLModel, table=False):
    search_term: str = Field(max_length=200)
    age_group: Optional[str] = Field(default=None, max_length=50)
    protocol_type: Optional[EmergencyType] = Field(default=None)


class MilestoneQuery(SQLModel, table=False):
    age_months: Optional[int] = Field(default=None)
    domain: Optional[DevelopmentalDomain] = Field(default=None)
    search_term: Optional[str] = Field(default=None, max_length=200)


class SymptomQuery(SQLModel, table=False):
    symptom_name: str = Field(max_length=200)
    age_months: Optional[int] = Field(default=None)
    additional_symptoms: Optional[List[str]] = Field(default=None)


class GrowthChartQuery(SQLModel, table=False):
    chart_type: str = Field(max_length=50)
    sex: str = Field(max_length=10)
    age_months: int
    measurement_value: Decimal = Field(decimal_places=2)
