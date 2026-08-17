import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class CaptureSession(Base):
    """One subject's recording day -- the fields config/session.yaml used to
    hold, minus `round`: a subject typically walks `total_rounds` times in a
    single day, and each round gets its own videos/jobs (see Job.round)
    rather than requiring a separate CaptureSession per round."""

    __tablename__ = "capture_sessions"

    id = Column(Integer, primary_key=True)
    subject_name = Column(String, nullable=False)
    day = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    p_no = Column(Integer, nullable=False)
    total_rounds = Column(Integer, nullable=False, default=1)
    camera_count = Column(Integer, nullable=False, default=2)
    input_dir = Column(String, nullable=False)
    output_dir = Column(String, nullable=False)
    alignment_method = Column(String, nullable=False, default="pca")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    jobs = relationship("Job", back_populates="capture_session", cascade="all, delete-orphan")


class Job(Base):
    """One subprocess run of a pipeline stage against a CaptureSession.
    `round` is which round of that session this job is for -- calibration
    jobs don't vary by round (calibration is shared across a whole recording
    day) so it's left null for those."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    capture_session_id = Column(Integer, ForeignKey("capture_sessions.id"), nullable=False)
    job_type = Column(String, nullable=False)  # calibration | pose_preview | pose_estimation
    round = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="pending")  # pending|running|succeeded|failed
    target_person_idx = Column(Integer, nullable=True)
    log_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    capture_session = relationship("CaptureSession", back_populates="jobs")


class GaitModel(Base):
    """An uploaded GaitGRU checkpoint (see app/gait_model.py)."""

    __tablename__ = "gait_models"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    train_subjects = Column(String, nullable=True)  # JSON-encoded list
    val_subject = Column(String, nullable=True)
    test_subject = Column(String, nullable=True)
    target_columns = Column(String, nullable=True)  # JSON-encoded list
    val_loss = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
